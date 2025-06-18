source ./.env
pip install -r requirements.txt
export PYTHONPATH="$REPO_PATH/conlid:$PYTHONPATH"