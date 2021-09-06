#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
just for rblib
"""

import sys
from distutils.core import setup,Extension

from Cython.Build import cythonize

package_list = ["rblib"]
package_dir = {"rblib": "rblib"}
#data_dir = {"rblib","rblib"}

#exts = cythonize([Extension("fib", sources=["cfib.c", "fib.pyx"])])

permutation = Extension("rblib.permutation.cal_permutation",
		[
			"rblib/permuatation/sr_msort.c",
			"rblib/permuatation/cal_permutation.c",
			])

scripts = ['scripts/SFnetx',]

metadata = {
		"name":"rblib",
		"version":"2.0.4",
		'description': "rblib",
		'long_description': __doc__,
		'author': "rongzhengqin",
		'author_email': "rongzhengqin@basepedia.com",
		'license': "MIT",
		'platforms': ["Linux","Mac OS-X","UNIX"],
		'url': "https://github.com/zju3351689/rblib",
		'packages': package_list,
		'package_dir': package_dir,
		'package_data': {'rblib': ['ref.types','ref.types.strands'],},
		'requires': ['cython (>=0.21)'],
		'ext_modules': cythonize("*/*.pyx",language_level=3) + [permutation,],
		'scripts':scripts,
		}

if __name__ == '__main__':
	dist = setup(**metadata)

