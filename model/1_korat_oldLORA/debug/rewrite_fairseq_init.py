import os
import site

# The CORRECTED content for fairseq/dataclass/initialize.py
# FIX: Uses 'cfg_name' instead of hardcoded string to ensure Hydra finds it.
new_content = """# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import MISSING
from fairseq.dataclass.configs import FairseqConfig
from hydra.core.config_store import ConfigStore

logger = logging.getLogger(__name__)

def hydra_init(cfg_name="config"):
    cs = ConfigStore.instance()
    
    # Iterate over all fields in the FairseqConfig to register sub-configs
    for k, v in FairseqConfig.__dataclass_fields__.items():
        node = None
        
        # 1. Check if there is a factory (The Python 3.11 Fix)
        if v.default_factory is not MISSING:
            try:
                # Instantiate the config class (e.g. CommonConfig())
                node = v.default_factory()
            except Exception as e:
                # logger.warning(f"Could not instantiate default_factory for {k}: {e}")
                pass
        
        # 2. Fallback to standard default if factory was missing
        if node is None and v.default is not MISSING:
            node = v.default
            
        # 3. If we still have nothing, skip it
        if node is None or node is MISSING:
            continue
            
        # 4. Register with Hydra
        try:
            cs.store(name=k, node=node)
        except Exception as e:
            pass

    # CRITICAL FIX: Register the main config using the variable 'cfg_name' (usually "config")
    # The previous script incorrectly named this "fairseq_config"
    cs.store(name=cfg_name, node=FairseqConfig)
"""

def overwrite_init():
    # Find the fairseq installation
    site_packages = site.getsitepackages()
    target_file = None
    
    for path in site_packages:
        potential_path = os.path.join(path, "fairseq", "dataclass", "initialize.py")
        if os.path.exists(potential_path):
            target_file = potential_path
            break
            
    if not target_file:
        print("❌ Could not find fairseq/dataclass/initialize.py")
        return

    print(f"🔧 Overwriting {target_file} with fixed naming logic...")
    
    with open(target_file, "w") as f:
        f.write(new_content)
        
    print("✅ Successfully restored initialize.py!")

if __name__ == "__main__":
    overwrite_init()
