#!/bin/bash
set -e

# Ensure Craftax texture caches exist in /cache
if [ ! -f /cache/craftax_assets/texture_cache.pbz2 ] || [ ! -f /cache/craftax_assets/texture_cache_classic.pbz2 ]; then
    echo "Generating Craftax texture caches..."
    mkdir -p /cache/craftax_assets
    python -c "
from craftax.craftax_env import make_craftax_env_from_name
make_craftax_env_from_name('Craftax-Symbolic-v1', True)
make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', True)
print('Craftax caches generated successfully.')
"
fi

# Run the user's command
exec "$@"
