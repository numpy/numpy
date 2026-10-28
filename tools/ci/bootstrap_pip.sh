#!/usr/bin/env bash
# Secure bootstrap for pip >= 26.2.2 (eMudhra/emSign CA incident, gh-33142)
set -euo pipefail
WHEEL="https://files.pythonhosted.org/packages/3f/06/\
3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/\
pip-26.2.2-py3-none-any.whl"
SHA="fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
python -m pip install "pip @ ${WHEEL}#sha256=${SHA}"
