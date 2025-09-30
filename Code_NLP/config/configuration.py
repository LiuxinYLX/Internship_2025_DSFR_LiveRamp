from dataclasses import dataclass, field
import torch

@dataclass
class DatasetConfig:
    project: str = "lranalytics-eu-660531"
    dataset: str = "crf_liveramp_data_science_work"
    
    raw_table: str = "LIUXIN_crf_product_join_0630"
    clean_table: str = "LIUXIN_crf_product_join_cleaned_0630"
    excluded_table: str = "LIUXIN_crf_product_join_excluded_0630"

    dataset_type: str = "supermarket" # or "cinema"
    key_cle: str = "country_id, barcode"
    main_barcode: str = "barcode"


@dataclass
class ModelConfig:
    project: str = "lranalytics-eu-660531"
    dataset: str = "crf_liveramp_data_science_work"
    
    embedding_path: str = "../sentence-transformers"
    model_save_path: str = "checkpoints/"
    
    batch_size: int = 16
    epochs: int = 100
    hidden_dim: int = 128

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    value_cols = None
    category_cols: list = field(default_factory=lambda:[
        "color_desc", 
        "size_desc"
    ])
    hierarchy_cols: list = field(default_factory=lambda: [
        "hierarchy_level1_desc",
        "global_hierarchy_level2_desc",
        "global_hierarchy_level3_desc",
        "global_hierarchy_level4_desc",
        "global_hierarchy_level5_desc",
        "global_hierarchy_level6_desc"
    ])
    # correct_hierarchy_cols: list = field(default_factory=lambda: [
    #    "correct_hierarchy_level1_desc",
    #    "correct_global_hierarchy_level2_desc",
    #    "correct_global_hierarchy_level3_desc",
    #    "correct_global_hierarchy_level4_desc",
    #    "correct_global_hierarchy_level5_desc",
    #    "correct_global_hierarchy_level6_desc"
    #])

    correct_hierarchy_cols: list = field(default_factory=lambda: [])
    
    label_cols: list = field(default_factory=lambda: [
        "is_error"
    ])