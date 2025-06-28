from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

class MisconceptionSimulator:
    """
    Simplified simulator for Kaggle Misconceptions pipeline.
    Handles loading data, calculating affinity scores, and training classifiers.
    """
    
    def __init__(self):
        """Initialize the misconception simulator with required models and data."""
        print("🚀 Initializing Misconceptions Simulator...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.misconceptions_db = self.load_misconceptions()
        self.tfidf = TfidfVectorizer(max_features=500, stop_words='english')
        print("✅ Simulator initialized successfully")
    
    def load_misconceptions(self):
        """
        Load real misconceptions database from CSV file.
        Falls back to sample data if file not found.
        
        Returns:
            pd.DataFrame: Misconceptions data with ID and name columns
        """
        try:
            df = pd.read_csv('data/misconception_mappingV3.csv')
            print(f"📚 Loaded {len(df)} misconceptions from CSV")
            return df
        except FileNotFoundError:
            print("⚠️  CSV not found, creating sample data...")
            return self.create_sample_misconceptions()
    
    def create_sample_misconceptions(self):
        """
        Create sample misconceptions data when CSV file is not available.
        
        Returns:
            pd.DataFrame: Sample misconceptions with common math errors
        """
        sample_data = {
            'MisconceptionId': [0, 1, 2, 3, 4],
            'MisconceptionName': [
                "Does not know that angles in a triangle sum to 180 degrees",
                "Confuses square with square root",
                "Does not apply inverse operations in correct order",
                "Confuses circumference and area formulas",
                "Adds percentage to number incorrectly"
            ]
        }
        return pd.DataFrame(sample_data)
    
    def generate_question_distractor_pairs(self, n_pairs=50):
        """
        Generate question-distractor pairs using real misconceptions.
        Creates realistic math problems with common wrong answers.
        
        Args:
            n_pairs (int): Number of question-distractor pairs to generate
            
        Returns:
            pd.DataFrame: Generated pairs with questions, distractors, and misconceptions
        """
        
        # Question templates based on common misconceptions
        question_templates = {
            "triangle_angles": [
                "If a triangle has angles 60° and 70°, what is the third angle?",
                "Find the missing angle in a triangle with angles 45° and 90°",
                "What is the third angle of a triangle if two angles are 30° and 60°?"
            ],
            "square_root": [
                "What is the square root of 64?",
                "Find √49",
                "Calculate the square root of 36"
            ],
            "algebra": [
                "If 2x + 5 = 15, what is x?",
                "Solve 3y - 7 = 14",
                "Find x when 4x + 8 = 20"
            ],
            "geometry": [
                "Find the area of a circle with radius 5 cm",
                "What is the circumference of a circle with radius 3 cm?",
                "Calculate the perimeter of a rectangle 8m × 5m"
            ],
            "percentage": [
                "What is 25% of 80?",
                "Calculate 15% of 200",
                "Find 30% of 150"
            ]
        }
        
        # Typical incorrect answers (distractors)
        wrong_answers = {
            "triangle_angles": ["60°", "90°", "120°"],
            "square_root": ["32", "12", "18"],
            "algebra": ["8", "3.5", "4"],
            "geometry": ["31.4", "15.7", "26"],
            "percentage": ["25", "35", "50"]
        }
        
        # Corresponding misconceptions (using real IDs from CSV)
        misconception_mapping = {
            "triangle_angles": 0,  # "Does not know that angles in a triangle sum to 180 degrees"
            "square_root": 1948,   # "Believes the inverse of square rooting is doubling"  
            "algebra": 2029,       # Similar to algebraic problems
            "geometry": 2072,      # "Confuses formulae for area of triangle and area of rectangle"
            "percentage": 1965     # "Thinks you need to just add a % sign to a decimal to convert to a percentage"
        }
        
        dataset = []
        categories = list(question_templates.keys())
        
        for i in range(n_pairs):
            category = random.choice(categories)
            question = random.choice(question_templates[category])
            distractor = random.choice(wrong_answers[category])
            
            # Search for corresponding misconception in database
            misconception_id = misconception_mapping.get(category, 0)
            misconception_row = self.misconceptions_db[
                self.misconceptions_db['MisconceptionId'] == misconception_id
            ]
            
            if not misconception_row.empty:
                misconception_name = misconception_row.iloc[0]['MisconceptionName']
            else:
                misconception_name = "Generic misconception"
            
            dataset.append({
                'question_id': f"Q{i+1:03d}",
                'question': question,
                'distractor': distractor,
                'misconception_id': misconception_id,
                'misconception_name': misconception_name,
                'category': category
            })
        
        return pd.DataFrame(dataset)
    
    def calculate_affinity_score(self, question: str, distractor: str) -> float:
        """
        Calculate affinity score between question and distractor.
        Higher scores indicate better question-distractor pairs.
        
        Args:
            question (str): The math question text
            distractor (str): The incorrect answer option
            
        Returns:
            float: Affinity score between 0 and 1
        """
        
        # 1. Semantic similarity with embeddings
        q_emb = self.model.encode([question])
        d_emb = self.model.encode([distractor])
        semantic_sim = cosine_similarity(q_emb, d_emb)[0][0]
        
        # 2. Simple lexical similarity (common words)
        q_words = set(question.lower().split())
        d_words = set(str(distractor).lower().split())
        
        if len(q_words.union(d_words)) > 0:
            lexical_sim = len(q_words.intersection(d_words)) / len(q_words.union(d_words))
        else:
            lexical_sim = 0
        
        # 3. Combined score
        affinity_score = 0.7 * semantic_sim + 0.3 * lexical_sim
        
        return affinity_score
    
    def filter_high_affinity_pairs(self, df: pd.DataFrame, threshold: float = 0.2) -> pd.DataFrame:
        """
        Filter pairs with high affinity scores (good distractors).
        
        Args:
            df (pd.DataFrame): Input dataframe with question-distractor pairs
            threshold (float): Minimum affinity score to keep pairs
            
        Returns:
            pd.DataFrame: Filtered dataframe with high affinity pairs only
        """
        print(f"🔍 Calculating affinities for {len(df)} pairs...")
        
        affinity_scores = []
        for _, row in df.iterrows():
            score = self.calculate_affinity_score(row['question'], row['distractor'])
            affinity_scores.append(score)
        
        df['affinity_score'] = affinity_scores
        
        # Filter pairs with high affinity
        high_affinity_df = df[df['affinity_score'] >= threshold].copy()
        
        print(f"✅ Filtered {len(high_affinity_df)}/{len(df)} pairs with affinity >= {threshold}")
        return high_affinity_df
    
    def train_misconception_classifier(self, df: pd.DataFrame):
        """
        Train a classifier to predict misconceptions from question-distractor pairs.
        
        Args:
            df (pd.DataFrame): Training data with questions, distractors, and misconception labels
            
        Returns:
            RandomForestClassifier or None: Trained classifier or None if training failed
        """
        print("🤖 Training misconception classifier...")
        
        if len(df) < 5:
            print("⚠️  Dataset too small to train classifier")
            return None
        
        # Prepare features
        questions = df['question'].tolist()
        distractors = df['distractor'].tolist()
        
        # Combine question and distractor as input
        combined_texts = [f"{q} [SEP] {d}" for q, d in zip(questions, distractors)]
        
        # Use TF-IDF for features
        X = self.tfidf.fit_transform(combined_texts)
        y = df['misconception_id'].values
        
        # Check if there are enough unique classes
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print("⚠️  Only one class available, cannot train classifier")
            return None
        
        # Split data (no stratification if few samples per class)
        test_size = min(0.3, 0.8)  # Adjust based on dataset size
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train Random Forest
        self.classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"📊 Classifier accuracy: {accuracy:.3f}")
        
        return self.classifier
    
    def predict_misconception(self, question: str, distractor: str) -> Tuple[int, str, float]:
        """
        Predict misconception for a question-distractor pair.
        
        Args:
            question (str): The math question
            distractor (str): The incorrect answer
            
        Returns:
            Tuple[int, str, float]: Misconception ID, name, and prediction confidence
        """
        if self.classifier is None:
            return -1, "Classifier not trained", 0.0
        
        combined_text = f"{question} [SEP] {distractor}"
        features = self.tfidf.transform([combined_text])
        
        prediction = self.classifier.predict(features)[0]
        probability = self.classifier.predict_proba(features)[0].max()
        
        # Find misconception name
        misconception_row = self.misconceptions_db[
            self.misconceptions_db['MisconceptionId'] == prediction
        ]
        
        if not misconception_row.empty:
            misconception_name = misconception_row.iloc[0]['MisconceptionName']
        else:
            misconception_name = "Unknown misconception"
        
        return prediction, misconception_name, probability
    
    def generate_analysis_report(self, df: pd.DataFrame):
        """
        Generate comprehensive analysis report of the dataset.
        
        Args:
            df (pd.DataFrame): Dataset to analyze with affinity scores and categories
        """
        print("\n" + "="*60)
        print("📈 ANALYSIS REPORT")
        print("="*60)
        
        print(f"📊 Total pairs processed: {len(df)}")
        print(f"🎯 Average affinity: {df['affinity_score'].mean():.3f}")
        print(f"📈 Maximum affinity: {df['affinity_score'].max():.3f}")
        print(f"📉 Minimum affinity: {df['affinity_score'].min():.3f}")
        
        print(f"\n🏷️  Categories found:")
        category_counts = df['category'].value_counts()
        for category, count in category_counts.items():
            print(f"   • {category}: {count} pairs")
        
        print(f"\n🧠 Top 5 Most common misconceptions:")
        misconception_counts = df['misconception_name'].value_counts().head()
        for i, (misconception, count) in enumerate(misconception_counts.items(), 1):
            print(f"   {i}. {misconception[:50]}... ({count} cases)")
    
    def run_simulation(self, n_pairs: int = 50):
        """
        Execute complete simulation pipeline.
        
        Args:
            n_pairs (int): Number of question-distractor pairs to generate
            
        Returns:
            pd.DataFrame: Final filtered dataset with high affinity pairs
        """
        print("\n" + "="*60)
        print("🎯 KAGGLE SIMULATION - MISCONCEPTIONS PIPELINE")
        print("="*60)
        
        # Step 1: Generate dataset
        df = self.generate_question_distractor_pairs(n_pairs)
        print(f"✅ Generated dataset with {len(df)} pairs")
        
        # Step 2: Filter by affinity
        df_filtered = self.filter_high_affinity_pairs(df, threshold=0.15)
        
        # Step 3: Train classifier
        classifier = self.train_misconception_classifier(df_filtered)
        
        # Step 4: Generate report
        self.generate_analysis_report(df_filtered)
        
        # Step 5: Show prediction examples
        if classifier is not None:
            print(f"\n🔮 PREDICTION EXAMPLES:")
            print("-" * 40)
            
            sample_pairs = df_filtered.head(3)
            for _, row in sample_pairs.iterrows():
                pred_id, pred_name, confidence = self.predict_misconception(
                    row['question'], row['distractor']
                )
                
                print(f"❓ Question: {row['question'][:50]}...")
                print(f"❌ Distractor: {row['distractor']}")
                print(f"🧠 Prediction: {pred_name[:40]}...")
                print(f"🎯 Confidence: {confidence:.3f}")
                print("-" * 40)
        
        return df_filtered

# Execute simulation
if __name__ == "__main__":
    simulator = MisconceptionSimulator()
    results = simulator.run_simulation(n_pairs=100)
    
    print(f"\n🎉 Simulation completed successfully!")
    print(f"📊 Final dataset: {len(results)} high-affinity pairs")
