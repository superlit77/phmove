# HMove: An Intelligent Refactoring Tool



## Quick start

### Installation

1. Environment: 

   ​	Windows 10;

   ​	Python3.9;

   ​	JDK-17.

2. Install dependencies from `requirements.txt` .

### Usage

1. Open a command-line window, and go to the hmove project directory.

2. Make sure all dependencies are installed.

3. In this tool, we use the method of loading pre-trained models online, so you need to configure the proxy environment variables before using them, for example:

   `set http_proxy=http://127.0.0.1:33210
   set https_proxy=http://127.0.0.1:33210`

4. You can invoke the following command to perform a single refactoring prediction in the command line window:

   `python new_main.py --move_file your\path\to\source\java\file --start_line 13 --moveto_file your\path\to\target\java\file`

   Note: After  `--start_line` , you need to fill in the number of starting lines of the method to be detected in the source file.

   Example:

   ​	`python new_main.py --move_file E:\cam\WCWidth.java --start_line 13 --moveto_file D:\ConsoleReader.java`

5. Intermediate files generated during program execution will be stored in the src directory for viewing.

### Use examples

![1718354852593](C:\Users\superlit77\AppData\Roaming\Typora\typora-user-images\1718354852593.png)



```
(hmove) E:\hmove\src>python new_main.py --move_file E:\cam\WCWidth.java --start_line 13 --moveto_file D:\ConsoleReader.java
Intermediate files will be stored in E:\hmove\src\output_2024-06-14_16-02-26
----------------------------------------------------------------------------
Start To Load Pretrain Model ... ...
Finish Loading Pretrain Model ! !
----------------------------------------------------------------------------
The operation is complete. All java files have been copied successfully.
Subprocess output: Parent path as string: output_2024-06-14_16-02-26
File created successfully: output_2024-06-14_16-02-26\relation.csv
... initializing  environment ...
Find Java File: E:\hmove\src\output_2024-06-14_16-02-26\source\ConsoleReader.java
Find Java File: E:\hmove\src\output_2024-06-14_16-02-26\source\WCWidth.java
... collecting  relations ...
119
Finish!

Subprocess finished successfully
Subprocess completed, continuing main program execution.
----------------------------------------------------------------------------
We recommend to move the method starting at line 13 in the source class to the target class.

```