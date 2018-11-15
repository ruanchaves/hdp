from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

def parallel_decorator(func):
	def wrapper(*args, **kwargs):
		with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
			executor.map(func, **args, **kwargs)
