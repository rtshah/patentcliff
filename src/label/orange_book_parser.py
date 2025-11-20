"""Parse Orange Book files (products, patents, exclusivity) and compute T0 events."""
import pandas as pd
from pathlib import Path
from datetime import date
from typing import Optional, Dict, List
import sys
from pathlib import Path as PathLib

# Add project root to path
project_root = PathLib(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import load_config, parse_date_flexible, find_column, normalize_column_name
from src.features.scd_normalize import extract_scd_from_product_row, parse_df_route


class OrangeBookParser:
    """Parse Orange Book files and extract events."""
    
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = load_config()
        self.config = config
        self.ob_dir = Path(config["paths"]["orange_book_dir"])
        self.use_exclusivity = config["features"]["use_exclusivity"]
    
    def load_products(self) -> pd.DataFrame:
        """Load products.txt and normalize columns."""
        products_path = self.ob_dir / "products.txt"
        if not products_path.exists():
            raise FileNotFoundError(f"Products file not found: {products_path}")
        
        # Read with tilde delimiter
        df = pd.read_csv(products_path, sep='~', dtype=str, keep_default_na=False)
        
        # Normalize column names
        column_mapping = {}
        
        # Map known aliases to standard names
        aliases = {
            "appl_no": ["Appl_No", "Application_Number", "APPL_NO"],
            "appl_type": ["Appl_Type", "Application_Type", "APPL_TYPE"],
            "product_no": ["Product_No", "Product_Number", "PRODUCT_NO"],
            "ingredient": ["Ingredient", "Active_Ingredient", "INGREDIENT"],
            "strength": ["Strength", "STRENGTH"],
            "df_route": ["DF;Route", "Dosage_Form;Route", "DF;ROUTE"],
            "trade_name": ["Trade_Name", "Proprietary_Name"],
            "te_code": ["TE_Code", "TE", "TE_CODE"],
            "approval_date": ["Approval_Date", "Appl_Approval_Date"],
        }
        
        for std_name, alias_list in aliases.items():
            found_col = find_column(df.columns.tolist(), alias_list)
            if found_col:
                column_mapping[found_col] = std_name
        
        df = df.rename(columns=column_mapping)
        
        # Parse dates
        if "approval_date" in df.columns:
            df["approval_date_parsed"] = df["approval_date"].apply(parse_date_flexible)
        else:
            df["approval_date_parsed"] = None
        
        # Extract SCD components
        scd_data = []
        for _, row in df.iterrows():
            scd_info = extract_scd_from_product_row(row.to_dict())
            scd_data.append(scd_info)
        
        scd_df = pd.DataFrame(scd_data)
        
        # Drop original ingredient and strength columns to avoid duplicates
        # (SCD extraction provides normalized versions we want to keep)
        columns_to_drop = []
        if "ingredient" in df.columns and "ingredient" in scd_df.columns:
            columns_to_drop.append("ingredient")
        if "strength" in df.columns and "strength" in scd_df.columns:
            columns_to_drop.append("strength")
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
        
        # Concatenate SCD data
        df = pd.concat([df, scd_df], axis=1)
        
        return df
    
    def load_patents(self) -> pd.DataFrame:
        """Load patent.txt and normalize columns."""
        patent_path = self.ob_dir / "patent.txt"
        if not patent_path.exists():
            raise FileNotFoundError(f"Patent file not found: {patent_path}")
        
        df = pd.read_csv(patent_path, sep='~', dtype=str, keep_default_na=False)
        
        # Map columns
        aliases = {
            "appl_no": ["Appl_No", "APPL_NO"],
            "patent_no": ["Patent_No", "PATENT_NO"],
            "patent_expire_date_text": ["Patent_Expire_Date_Text", "PATENT_EXPIRE_DATE_TEXT"],
            "patent_expire_date": ["Patent_Expire_Date", "PATENT_EXPIRE_DATE"],
        }
        
        column_mapping = {}
        for std_name, alias_list in aliases.items():
            found_col = find_column(df.columns.tolist(), alias_list)
            if found_col:
                column_mapping[found_col] = std_name
        
        df = df.rename(columns=column_mapping)
        
        # Parse expiry dates
        if "patent_expire_date_text" in df.columns:
            df["patent_expire_date_parsed"] = df["patent_expire_date_text"].apply(parse_date_flexible)
        elif "patent_expire_date" in df.columns:
            df["patent_expire_date_parsed"] = df["patent_expire_date"].apply(parse_date_flexible)
        else:
            df["patent_expire_date_parsed"] = None
        
        return df
    
    def load_exclusivity(self) -> Optional[pd.DataFrame]:
        """Load exclusivity.txt if available and enabled."""
        if not self.use_exclusivity:
            return None
        
        excl_path = self.ob_dir / "exclusivity.txt"
        if not excl_path.exists():
            return None
        
        df = pd.read_csv(excl_path, sep='~', dtype=str, keep_default_na=False)
        
        aliases = {
            "appl_no": ["Appl_No", "APPL_NO"],
            "exclusivity_code": ["Exclusivity_Code", "EXCLUSIVITY_CODE"],
            "exclusivity_date": ["Exclusivity_Date", "EXCLUSIVITY_DATE"],
        }
        
        column_mapping = {}
        for std_name, alias_list in aliases.items():
            found_col = find_column(df.columns.tolist(), alias_list)
            if found_col:
                column_mapping[found_col] = std_name
        
        df = df.rename(columns=column_mapping)
        
        if "exclusivity_date" in df.columns:
            df["exclusivity_date_parsed"] = df["exclusivity_date"].apply(parse_date_flexible)
        else:
            df["exclusivity_date_parsed"] = None
        
        return df
    
    def compute_t0_events(self) -> pd.DataFrame:
        """
        Compute T0 (loss of protection) events per NDA SCD.
        Returns DataFrame with columns: appl_no, scd_key, ingredient, strength, 
        dosage_form, route, t0, used_exclusivity
        """
        products = self.load_products()
        patents = self.load_patents()
        exclusivity = self.load_exclusivity()
        
        # Filter to NDA products only
        nda_products = products[products["appl_type"].str.upper() == "N"].copy()
        
        if nda_products.empty:
            raise ValueError("No NDA products found in Orange Book")
        
        # Join patents to products
        nda_products = nda_products.merge(
            patents[["appl_no", "patent_expire_date_parsed"]],
            on="appl_no",
            how="left"
        )
        
        # Group by (appl_no, scd_key) and compute max patent expiry
        patent_max = nda_products.groupby(["appl_no", "scd_key"]).agg({
            "patent_expire_date_parsed": "max",
            "ingredient": "first",
            "strength": "first",
            "dosage_form": "first",
            "route": "first",
            "dosage_form_canonical": "first",
            "route_canonical": "first",
        }).reset_index()
        
        patent_max = patent_max.rename(columns={"patent_expire_date_parsed": "last_patent_expire"})
        
        # If exclusivity is enabled, join and compute max
        if exclusivity is not None and not exclusivity.empty:
            # Merge exclusivity (may be at appl_no level only)
            excl_max = exclusivity.groupby("appl_no")["exclusivity_date_parsed"].max().reset_index()
            excl_max = excl_max.rename(columns={"exclusivity_date_parsed": "last_exclusivity_expire"})
            
            patent_max = patent_max.merge(excl_max, on="appl_no", how="left")
            
            # T0 = max(patent, exclusivity)
            patent_max["t0"] = patent_max[["last_patent_expire", "last_exclusivity_expire"]].max(axis=1)
            patent_max["used_exclusivity"] = (
                patent_max["last_exclusivity_expire"].notna() &
                (patent_max["last_exclusivity_expire"] >= patent_max["last_patent_expire"].fillna(date.min))
            )
        else:
            patent_max["t0"] = patent_max["last_patent_expire"]
            patent_max["used_exclusivity"] = False
        
        # Filter to events with valid T0 in our time window
        events = patent_max[patent_max["t0"].notna()].copy()
        
        # Drop future dates (beyond reasonable range)
        max_reasonable_date = date(2035, 12, 31)
        events = events[events["t0"] <= max_reasonable_date]
        
        return events[["appl_no", "scd_key", "ingredient", "strength", "dosage_form", 
                      "route", "dosage_form_canonical", "route_canonical", "t0", "used_exclusivity"]]

