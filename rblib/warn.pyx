import sys
import os


class Warn(object):
	def __init__(self,logfile=sys.stderr,level=2): # 1 is report Error + Warn , 2 is report report Error + Warn + Info , 0 is report Error
		self.level = level
		self.logfile = logfile
		self.error = {
				"ERROR" : 0,
				"WARN"  : 1,
				"INFO"  : 2,
				}
	def info(self,errorlevel,infomation):
		if self.error[errorlevel] <= self.level:
			self.logfile.write("[%s] %s.%s"%(errorlevel,infomation,os.linesep))


if __name__ == "__main__":
	log = Warn()
	log.info("ERROR","hello")

