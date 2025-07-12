import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import requests
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# === CONFIGURATION ===
# Using local model with Ollama

INPUT_TRAIN = "data/train.csv"
INPUT_MAPPING = "data/misconception_mappingV1.csv"
OUTPUT_PATH = "output/results.csv"
AFFINITY_THRESHOLD = 0.15
MIN_MISCONCEPTIONS_PER_DISTRACTOR = 25

# Testing configuration - set to None to process all distractors
MAX_DISTRACTORS_FOR_TESTING = 20  # Change to None for full processing

# === FUNCTIONS ===

def load_and_merge_data(train_path, mapping_path):
    df_train = pd.read_csv(train_path)
    df_map = pd.read_csv(mapping_path)

    rows = []
    for _, row in df_train.iterrows():
        for opt in ['A', 'B', 'C', 'D']:
            if pd.notna(row[f'Answer{opt}Text']):
                rows.append({
                    "question": row["QuestionText"],
                    "distractor": row[f"Answer{opt}Text"],
                    "misconception_id": row[f"Misconception{opt}Id"]
                })

    df_long = pd.DataFrame(rows)
    df_final = df_long.merge(df_map, how='left', left_on='misconception_id', right_on='MisconceptionId')
    return df_final.dropna(subset=["distractor"])

def filter_by_affinity(df, threshold=0.15):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    similarities = []
    for _, row in df.iterrows():
        sim = util.cos_sim(
            model.encode(row['question'], convert_to_tensor=True),
            model.encode(row['distractor'], convert_to_tensor=True)
        ).item()
        similarities.append(sim)
    df['affinity'] = similarities
    return df[df['affinity'] >= threshold]

def call_local_llm_misconception_label(question, distractor):
    prompt = f"""
Question: {question}
Distractor (incorrect answer): {distractor}

What mathematical misconception could explain why someone chose this incorrect answer?
Respond briefly and clearly.
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3", 
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error connecting to local model: {str(e)}"

def validate_with_local_llm(question, distractor, prediction):
    prompt = f"""
Question: {question}
Distractor (incorrect answer): {distractor}
Misconception prediction: "{prediction}"

Does this prediction make sense as a conceptual error related to the question and distractor? 
Respond only "yes" or "no", followed by a brief justification.
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3", 
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            content = response.json()["response"].lower()
            return "yes" in content or "sí" in content
        else:
            return False
    except Exception as e:
        print(f"Validation error: {str(e)}")
        return False

# === TEST FUNCTION ===

def test_local_llm():
    """Test if the local model is working"""
    print("🧪 Testing connection with local model...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama working. Available models: {[m['name'] for m in models]}")
            
            # Quick test
            test_response = call_local_llm_misconception_label(
                "What is the square root of 64?", 
                "12"
            )
            print(f"🤖 Test response: {test_response}")
            return True
        else:
            print(f"❌ Connection error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {str(e)}")
        print("💡 Make sure Ollama is running and the 'gemma3' model is installed")
        return False

# === MAIN WORKFLOW ===

def main():
    # First test connection with local model
    if not test_local_llm():
        print("❌ Cannot continue without the local model working")
        return
    
    print("\n🔍 Loading data...")
    df = load_and_merge_data(INPUT_TRAIN, INPUT_MAPPING)
    df_filtered = filter_by_affinity(df, threshold=AFFINITY_THRESHOLD)
    print(f"✅ Filtered {len(df_filtered)} pairs")
    
    # Group by distractor to ensure we get 25+ misconceptions per distractor
    distractor_groups = df_filtered.groupby('distractor')
    total_distractors = len(distractor_groups)
    
    # Determine how many distractors to process
    if MAX_DISTRACTORS_FOR_TESTING is None:
        distractors_to_process = total_distractors
        print(f"📊 Found {total_distractors} unique distractors")
        print(f"🚀 Processing ALL distractors (full production run)")
    else:
        distractors_to_process = min(MAX_DISTRACTORS_FOR_TESTING, total_distractors)
        print(f"📊 Found {total_distractors} unique distractors")
        print(f"🧪 TESTING MODE: Processing only first {distractors_to_process} distractors")
    
    all_results = []
    distractor_count = 0
    
    for distractor, group in distractor_groups:
        distractor_count += 1
        
        # Stop after processing the desired number of distractors
        if MAX_DISTRACTORS_FOR_TESTING and distractor_count > MAX_DISTRACTORS_FOR_TESTING:
            break
            
        print(f"\n🎯 Processing distractor {distractor_count}/{distractors_to_process}")
        print(f"📝 Distractor: {distractor[:60]}...")
        print(f"🔢 {len(group)} questions with this distractor")
        
        # Process up to 25 examples for this distractor (or all if less than 25)
        sample_size = min(MIN_MISCONCEPTIONS_PER_DISTRACTOR, len(group))
        group_sample = group.head(sample_size)
        
        misconceptions = []
        validated = []
        
        for idx, (_, row) in enumerate(group_sample.iterrows(), 1):
            print(f"\n  ➡️ Processing {idx}/{sample_size}: {row['question'][:40]}...")
            
            # Generate misconception
            label = call_local_llm_misconception_label(row['question'], row['distractor'])
            print(f"  🤖 LLM: {label[:80]}...")
            
            # Validate misconception
            valid = validate_with_local_llm(row['question'], row['distractor'], label)
            print(f"  ✅ Validated: {valid}")
            
            misconceptions.append(label)
            validated.append(valid)
        
        # Add results to the group
        group_sample = group_sample.copy()
        group_sample['predicted_misconception'] = misconceptions
        group_sample['llm_validated'] = validated
        
        # Filter only validated ones
        validated_results = group_sample[group_sample['llm_validated'] == True]
        all_results.append(validated_results)
        
        print(f"  📈 Validated {len(validated_results)} of {len(group_sample)} for this distractor")
    
    # Combine all results
    if all_results:
        df_final = pd.concat(all_results, ignore_index=True)
        
        # Create output directory
        os.makedirs("output", exist_ok=True)
        df_final.to_csv(OUTPUT_PATH, index=False)
        
        print(f"\n📁 Results saved to {OUTPUT_PATH}")
        print(f"📊 Total validated misconceptions: {len(df_final)}")
        
        # Summary by distractor
        summary = df_final.groupby('distractor').size().reset_index(name='count')
        print("\n📋 Summary by distractor:")
        for _, row in summary.head(10).iterrows():
            print(f"  • {row['distractor'][:50]}...: {row['count']} misconceptions")
    else:
        print("❌ No validated results found")

if __name__ == "__main__":
    main()
