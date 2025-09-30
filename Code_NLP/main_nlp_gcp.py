import pickle
import pandas as pd
from google.cloud import bigquery

from modules.generateEmbedding import generate_embedding
from modules.nlp import NLPDataset
from modules.model import (
    train_model, 
    inference_model,
    decode_predictions
)
from config.utils import do_data2table_job
from config.configuration import ModelConfig


def run(cfg: ModelConfig,
        client: bigquery.Client,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        mode: str,
        load_from_checkpoint: bool = True,
        model_read_name = "final_model.pt",
        model_save_name = "final_model.pt",
        return_levels: bool = False):
    
    # 1.Check dataset
    print("Step 1: Checking datasets...")
    if train_df is None and test_df is None:
        raise ValueError("Both train_df and test_df cannot be None. Please provide at least one DataFrame.")
    if train_df is None:
        train_df = test_df
    if test_df is None:
        test_df = train_df

    # 2.Load dataset and generate embeddings
    print("Step 2: Loading datasets and generating embeddings...")
    train_df['split'] = 'train'
    test_df['split'] = 'test'
    df = pd.concat([train_df, test_df], ignore_index=True)
    df["description"] = "Description du produit: " + df["item_desc"] + "local_brand_name: " + df["local_brand_name"] + ", global_brand_name: " + df["global_brand_name"]
        
    if mode == 'train':
        print("Training mode: Training the model...")
        X_all, y_all, ylabels_all, label_all, label_map, reverse_label_map= generate_embedding(
            cfg=cfg,
            df=df,
            model_path=cfg.embedding_path,
            text_cols="description",
            include_y_in_x=True,
            existing_label_map=None
        )
    
        with open('checkpoints/label_map.pkl', 'wb') as f:
            pickle.dump((label_map, reverse_label_map), f)
        # dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        split_mask = df['split'] == 'train'
        X_train, y_train, ylabels_train, label_train = X_all[split_mask], y_all[split_mask], ylabels_all[split_mask], label_all[split_mask]
        train_dataset = NLPDataset(X_train, y_train, ylabels_train, label_train, return_levels=return_levels)
        # n_classes_per_level = [len(df[col].unique()) for col in cfg.hierarchy_cols] + [2]  # +2 for the is_error label (0 and 1)
        n_classes_per_level = [len(label_map[col]) for col in cfg.hierarchy_cols] + [2]
        
        train_model(
            cfg = cfg,
            dataset = train_dataset,
            input_dim=X_train.shape[1],
            n_classes_per_level=n_classes_per_level,
            model_read_name = model_read_name,
            model_save_name = model_save_name,
            load_from_checkpoint = load_from_checkpoint,
            return_levels = return_levels
        )
    # Inference
    elif mode == 'inference':
        print("Inference mode: Loading the model for inference...")
        pd.set_option('display.max_columns', None)

        with open('checkpoints/label_map.pkl', 'rb') as f:
            label_map, reverse_label_map = pickle.load(f)


        X_all, y_all, ylabels_all, label_all, _, _ = generate_embedding(
            cfg=cfg,
            df=df,
            model_path=cfg.embedding_path,
            text_cols="description",
            include_y_in_x=True,
            existing_label_map=label_map
        )
            
        split_mask = df['split'] == 'train'
        X_test, y_test, ylabels_test, label_test = X_all[~split_mask], y_all[~split_mask], ylabels_all[~split_mask], label_all[~split_mask]
        test_dataset = NLPDataset(X_test, y_test, ylabels_test, label_test, return_levels=return_levels)
        n_classes_per_level = [len(label_map[col]) for col in cfg.hierarchy_cols] + [2]

        pred_labels,pred_probs = inference_model(
            cfg = cfg,
            dataset = test_dataset,
            input_dim = X_test.shape[1],
            n_classes_per_level = n_classes_per_level,
            model_read_name = model_read_name,
            return_levels = return_levels
        )

        if return_levels:
            decode_cols = cfg.hierarchy_cols + cfg.label_cols
            decoded_df = decode_predictions(pred_labels, reverse_label_map, decode_cols, test_df)
            decoded_df.to_csv("data/decoded_predictions_v3.csv", index=True)
        else:
            test_df = test_df.reset_index(drop=True)
            test_df['pred_is_error'] = pred_labels
            error = test_df[test_df["is_error"] != test_df["pred_is_error"]]
            test_df = test_df.reset_index(drop=True)
            test_df['pred_is_error'] = pred_labels
            test_df['pred_probs'] = pred_probs
            error = test_df[test_df["is_error"] != test_df["pred_is_error"]]
            
            force_string = [
                "hierarchy_level1_desc","global_hierarchy_level2_desc","global_hierarchy_level3_desc",
                "global_hierarchy_level4_desc","global_hierarchy_level5_desc","global_hierarchy_level6_desc",
                "local_brand_name","global_brand_name","item_desc","color_desc","size_desc",
                "barcode","country_id",
            ]

            schema=[
                bigquery.SchemaField("is_error", "BOOLEAN"),
                bigquery.SchemaField("pred_is_error", "BOOLEAN"),
                bigquery.SchemaField("split", "STRING"),
            ] + [bigquery.SchemaField(c, "STRING") for c in force_string]


            if not error.empty:
                print(error)
                do_data2table_job(cfg, client, None, "LIUXIN_crf_nlp_suspects_errors", error, schema)
            else:
                top_10_suspects = test_df.sort_values(by='pred_probs', ascending=False).head(10)
                print("Top 10 suspect errors based on predicted error probability:")
                print(top_10_suspects)
                do_data2table_job(cfg, client, None, "LIUXIN_crf_nlp_suspects_errors", top_10_suspects, schema)

            
        print("\n Les résultats sont enregistrés dans le tableau LIUXIN_crf_nlp_suspects_errors de BigQuery.")

    else:
        raise ValueError("Invalid mode. Choose 'train' or 'inference'.")



cfg = ModelConfig()
client = bigquery.Client()
train_df = client.query("SELECT * FROM `lranalytics-eu-660531.crf_liveramp_data_science_work.LIUXIN_crf_nlp_train_small`").to_dataframe()
test_df = client.query("SELECT * FROM `lranalytics-eu-660531.crf_liveramp_data_science_work.LIUXIN_crf_nlp_test_small`").to_dataframe()

result = run(
    cfg,
    client,
    train_df=train_df,
    test_df=test_df,
    mode='inference',
    load_from_checkpoint=False,
    model_save_name = "model_only_label_0922_gcp.pt",
    model_read_name = "model_only_label_0922_gcp.pt",
    return_levels=False
)