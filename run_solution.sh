# Runs solution on data in data/processed/ folder
SOLUTION_NAME=$1
pipenv run python3 solutions/$SOLUTION_NAME/score_model.py
