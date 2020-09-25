cmake_minimum_required( VERSION 2.8 )

project(2dparametricbem)

set(DIRECTORIES src test examples doxygen)
foreach(directory ${DIRECTORIES})
	add_subdirectory(${directory})
endforeach(directory)

