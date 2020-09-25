# A 2D Boundary Element Code using Transformation-Based Quadrature

* This code implements the approach outlined in Section 1.4.3.4 of the course Advanced Numerical Methods for CSE which describes various transformations to evaluate Galerkin Matrices using parametric curves. 

* Lead developer: Piyush Panchal

## Coding style

* Names should be descriptive, avoiding abbreviations. Common abbreviations can be used e.g. "num"
	* File Names
		* Should be all lowercase. Underscores "_" can be included in the file name.

	* Type Names
		* Type names start with a capital letter and have a capital letter for each new word with no underscores.
		* This applies for names of all types - classes, structs, type aliases, enums and type template parameters

	* Variable names
		* The names of variables and data members are all lowercase, with underscores between words.
		* Data members of classes (but not structs) additionally have trailing underscores.

	* Constant names
		* Variables declared constexpr or const, and whose value is fixed for the duration of the program, are named with a leading "k" followed by mixed case.

	* Namespace name
		* Namespace names are all lower-case. Top-level namespace names are based on the project name . Avoid collisions between nested namespaces and well-known top-level namespaces. Abbreviations not allowed for namespace names.

* Namespace usage
	* Follow the rules on Namespace Names.
	* Terminate namespaces with comments '// namespace name_of_namespace'.
	* Namespaces wrap the entire source file after includes, gflags definitions/declarations and forward declarations of classes from other namespaces.
	* You may not use a using-directive to make all names from a namespace available.

* Header files
	* All of a project's header files should be listed as descendants of the project's source directory without use of UNIX directory shortcuts . (the current directory) or .. (the parent directory).
 	* All header files shoud have #define guards to prevent multiple inclusion
	* Avoid forward declarations where possible. #include the headers required
	* Order of includes : Related header, C library, C++ library, other libraries, project's header. A blank line between Related header files and C library header file & C++ library header files and other library's header files.

* Declaration order
	* A class definition should usually start with a public: section, followed by protected:, then private:. Omit sections that would be empty.

* File Comments
 	* If a .h declares multiple abstractions, the file-level comment should broadly describe the contents of the file, and how the abstractions are related. A 1 or 2 sentence file-level comment may be sufficient. The detailed documentation about individual abstractions belongs with those abstractions, not at the file level.

* Class Comments
	* Every non-obvious class declaration should have an accompanying comment that describes what it is for and how it should be used.

* Function Comments
	* Declaration comments describe use of the function (when it is non-obvious); comments at the definition of a function describe operation.
	* Function Declarations : Almost every function declaration should have comments immediately preceding it that describe what the function does and how to use it.
	* Function Definitions : If there is anything tricky about how a function does its job, the function definition should have an explanatory comment.

* Spaces vs. Tabs
	* Use only spaces, and indent 2 spaces at a time.

## Comments

* Code commented using Doxygen; all doxygen related comments must be inside /* */
  (C-style comments)
* For comments to be typeset with LaTeX use C++-style comments // ....
  These comments should be used inside methods/functions in order to explain 
  details of the algorithms.

## Directory structure

root<br/>
&nbsp;&nbsp;|- doxygen<br/>
&nbsp;&nbsp;|- include<br/>
&nbsp;&nbsp;|- src<br/>
&nbsp;&nbsp;|&nbsp;&nbsp;|- Parametrizations<br/>
&nbsp;&nbsp;|&nbsp;&nbsp;|- Quadrature<br/>
&nbsp;&nbsp;|- test

## Demos and tests

* Directory "test" contains tests.cpp which contains unit tests for different parts of this project.
* Build the test executable from the build directory using the command "make parametricbem2d_tests".

## Documentation and examples

* Directory "doxygen" contains the Doxygen configuration file for generating the documentation which can be done by invoking the command "make doxygen_parametricbem2d" from the build folder.
* For getting started with examples, refer to "README.md" in "examples" directory.
