import os

# list all approaches available
__all__ = list(
    map(lambda x: x[:-3],
        filter(lambda x: x not in ['__init__.py', 'incremental_learning.py'] and x.endswith('.py'),
               os.listdir(os.path.dirname(__file__))
               )
        )
)
