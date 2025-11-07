import pandas as pd
from agentsearch.dataset.questions import Question, questions_df
from agentsearch.dataset.agents import Agent
from tqdm import tqdm
import sys

def main():
    print(f"Total questions available: {len(questions_df)}")

    sample_size = min(100, len(questions_df))
    sampled_questions_df = questions_df.sample(n=sample_size, random_state=42)

    print(f"Sampling {sample_size} questions for semantic matching and grading...")
    print(f"Question IDs: {sampled_questions_df.index.tolist()[:10]}...")

    results = []
    collection = "agents"

    for idx in tqdm(sampled_questions_df.index, desc="Grading questions"):
        question = Question.from_id(idx)

        try:
            matches = Agent.match(question, top_k=1, collection=collection)
            if not matches:
                print(f"\nNo semantic match found for question {idx}", file=sys.stderr)
                results.append({
                    'question_id': question.id,
                    'agent_id': None,
                    'agent_name': None,
                    'semantic_distance': None,
                    'question_text': question.text[:100],
                    'grade': None
                })
                continue

            agent_match = matches[0]
            agent = agent_match.agent
            grade = agent.grade(question)

            results.append({
                'question_id': question.id,
                'agent_id': agent.id,
                'agent_name': agent.name,
                'semantic_distance': agent_match.distance,
                'question_text': question.text[:100],
                'grade': grade
            })
        except Exception as e:
            print(f"\nError grading question {idx}: {e}", file=sys.stderr)
            results.append({
                'question_id': question.id,
                'agent_id': None,
                'agent_name': None,
                'semantic_distance': None,
                'question_text': question.text[:100],
                'grade': None
            })

    results_df = pd.DataFrame(results)
    output_file = 'results/graded_sampled_questions.csv'
    results_df.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print(f"Grading complete! Results saved to: {output_file}")
    print(f"{'='*60}")
    print("\nGrade Distribution:")
    print(results_df['grade'].value_counts().sort_index())
    print(f"\nAverage grade: {results_df['grade'].mean():.2f}")
    print(f"Median grade: {results_df['grade'].median():.1f}")
    print(f"\nAverage semantic distance: {results_df['semantic_distance'].mean():.4f}")
    print(f"\nSample results:")
    print(results_df[['question_id', 'agent_name', 'semantic_distance', 'grade']].head(10))

if __name__ == "__main__":
    main()
