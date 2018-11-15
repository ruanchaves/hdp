import multiprocessing

class Test(object):

	def __init__(self):
		self.manager = multiprocessing.Manager()
		self.list = self.manager.list()

	def method(self, i ):
		self.list.append([123, i])

	def run(self):
		pr = [0] * 10**6
		for i in range(0,10**6):
			pr[i] = multiprocessing.Process(target=self.method, args = (i,) )
			pr[i].start()
		for i in range(0,10**6):
			pr[i].join()
	

class Runner(object):
	def __init__(self):
		self.t = Test()

	def run(self):
		pr1 = multiprocessing.Process(target=self.t.method)
		pr1.start()
		pr1.join()
		print(self.t.list)

r = Test()
r.run()
