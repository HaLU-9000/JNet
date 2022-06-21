import sys
sys.path.append('/home/haruhiko/Documents/JNet')
from gomi import Gomi
path = 'dataset_test'
a = Gomi(path)
a.save()
