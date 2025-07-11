

from pathlib import Path

#print(f"Model will load from: {Path(__file__).parent / 'best_model.h5'}")

from pathlib import Path

print(Path(__file__))          # Output: PosixPath('/home/project/backend/cropAi/views.py')
print(Path(__file__).parent)   # Output: PosixPath('/home/project/backend/cropAi')