"""
A data organization and compilation system.
"""

from datamate.directory import (
    Directory,
    ArrayFile,
    set_root_dir,
    get_root_dir,
    enforce_config_match,
    check_size_on_init,
    get_check_size_on_init,
    set_verbosity_level,
    root,
    set_root_context,
    reset_scope,
)
from datamate.namespaces import Namespace, namespacify
from datamate.version import __version__

__all__ = ["ArrayFile", "Directory", "Namespace", "__version__"]

# -- `__module__` rebinding ----------------------------------------------------

Directory.__module__ = __name__
ArrayFile.__module__ = __name__
Namespace.__module__ = __name__
