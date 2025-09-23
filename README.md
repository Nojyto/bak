python3 -m venv --system-site-packages .venv
source .venv/bin/activate

pip-compile requirements-dev.in
pip-sync requirements.txt requirements-dev.txt

pip-compile requirements.in
pip-sync requirements.txt
