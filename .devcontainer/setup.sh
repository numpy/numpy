#!/usr/bin/env bash
set -euxo pipefail
if ! command -v micromamba &>/dev/null; then
    "${SHELL}" <(curl -Ls https://micro.mamba.pm/install.sh) < /dev/null
fi
micromamba shell init -s bash -p ~/micromamba
if [ -f environment.yml ]; then
    micromamba env create -f environment.yml --yes || micromamba env update -f environment.yml --yes
else
    exit 1
fi
git submodule update --init --recursive
cat > /opt/conda/.condarc <<'EOF'
envs_dirs:
  - /home/codespace/micromamba/envs
pkgs_dirs:
  - /home/codespace/micromamba/pkgs
EOF
echo "Setup completed successfully!"
echo "Run 'micromamba activate numpy-dev' to enter the environment."
