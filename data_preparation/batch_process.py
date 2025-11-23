"""
Batch processing script to fetch treatment recommendations.
Reads patient complaints from CSV and retrieves relevant treatment guidelines
using RAG (no model inference needed).
"""

import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def batch_process_pipeline(
    input_csv_path: str,
    output_csv_path: str,
    recommendation_rag_index_path: str = "models/recommendation_rag_index.joblib",
):
    """
    Fetch treatment recommendations for patient complaints using RAG.
    
    Args:
        input_csv_path: Path to input CSV with columns: Symptoms_comment, Causes_and_Disease, Medicine_recommendation
        output_csv_path: Path to save output CSV with: Symptoms_comment, Cause_and_Disease, Fetched_recommendation
        recommendation_rag_index_path: Path to recommendation RAG index
    """
    
    print("=" * 60)
    print("BATCH PROCESSING - RAG RETRIEVAL ONLY")
    print("=" * 60)
    
    # Load input data
    print(f"\n[1/3] Loading input data from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    print(f"Loaded {len(df)} patient complaints")
    
    # Verify required columns
    required_columns = ['Symptoms_comment', 'Causes_and_Disease']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV must have columns: {required_columns}. Missing: {missing_cols}")
    
    # Load RAG index
    print(f"\n[2/3] Loading RAG index from: {recommendation_rag_index_path}")
    index = joblib.load(recommendation_rag_index_path)
    diseases = index["diseases"]
    treatments = index["treatments"]
    treatment_embeddings = np.array(index["treatment_embeddings"], dtype=np.float32)
    
    print("  - Loading embedding model...")
    embedder = SentenceTransformer(index["embedding_model_name"])
    print("✓ RAG index loaded successfully!")
    
    # Process each complaint
    print(f"\n[3/3] Processing {len(df)} patient complaints...")
    
    results = []
    errors = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Fetching recommendations"):
        try:
            symptom = str(row['Symptoms_comment']).strip()
            cause_disease = str(row['Causes_and_Disease']).strip()
            
            # Skip invalid rows
            if not symptom or symptom.lower() == 'nan':
                errors.append({
                    'index': idx,
                    'symptom': symptom,
                    'error': 'Empty or invalid symptom'
                })
                continue
            
            if not cause_disease or cause_disease.lower() == 'nan':
                errors.append({
                    'index': idx,
                    'symptom': symptom,
                    'error': 'Empty or invalid cause/disease'
                })
                continue
            
            # Build retrieval query
            retrieval_query = f"Symptoms: {symptom}\nCause/Disease: {cause_disease}"
            
            # Retrieve top 1 treatment guideline using RAG
            q_emb = embedder.encode(
                [retrieval_query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0].astype(np.float32)
            
            # Compute cosine similarities
            sims = np.dot(treatment_embeddings, q_emb)
            top_idx = np.argmax(sims)
            
            # Extract top 1 fetched recommendation
            top_disease = diseases[top_idx]
            top_treatment = treatments[top_idx]
            fetched_recommendation = f"Disease: {top_disease}\nTreatment: {top_treatment}"
            
            # Store results
            results.append({
                'Symptoms_comment': symptom,
                'Cause_and_Disease': cause_disease,
                'Fetched_recommendation': fetched_recommendation,
            })
            
        except Exception as e:
            errors.append({
                'index': idx,
                'symptom': symptom if 'symptom' in locals() else 'Unknown',
                'error': str(e)
            })
            print(f"\n⚠ Error processing row {idx}: {e}")
            continue
    
    # Save results
    print(f"\n[4/4] Saving results to: {output_csv_path}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"✓ Saved {len(results)} processed results")
    
    # Report errors if any
    if errors:
        print(f"\n⚠ {len(errors)} errors occurred during processing:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - Row {error['index']}: {error['error']}")
        
        # Save errors to separate file
        errors_csv = output_csv_path.replace('.csv', '_errors.csv')
        errors_df = pd.DataFrame(errors)
        errors_df.to_csv(errors_csv, index=False)
        print(f"✓ Error details saved to: {errors_csv}")
    
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Successfully processed: {len(results)}/{len(df)} complaints")
    print(f"Output saved to: {output_csv_path}")
    

def main():
    """
    Main function to run batch processing.
    Update the input/output paths as needed.
    """
    # Configuration
    INPUT_CSV = "data/symptom_cause_disease_medicine_transformed_varied.csv"  # Update this path
    OUTPUT_CSV = "data/processed_results.csv"  # Update this path
    
    # Run batch processing
    batch_process_pipeline(
        input_csv_path=INPUT_CSV,
        output_csv_path=OUTPUT_CSV,
    )


if __name__ == "__main__":
    import sys
    
    # Allow command-line arguments for input/output paths
    if len(sys.argv) >= 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        print(f"Using custom paths:")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        batch_process_pipeline(input_path, output_path)
    else:
        # Use default paths
        main()
