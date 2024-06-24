# An HMove-based command-line tool

**This is a multi-tool integrated mobile method reconstruction platform.** We have integrated HMove and FeTruth into a command-line tool, streamlining the process for developers to reproduce results from other move method refactoring tools. This integration facilitates users in obtaining refactoring recommendations from various tools with ease.



## Quick start

### Installation

1. Environment: 

   ​	Windows 10;

   ​	Python3.9;

   ​	JDK-17.

2. Install dependencies from `requirements.txt` .



## Usage

### Usage of HMove tool

1. Open a command-line window, and go to the hmove_tool project directory. (phmove->tool-> hmove_tool)

2. Make sure all dependencies are installed.

3. **In hmove, we use the method of loading pre-trained models online, so you need to configure the proxy environment variables before using them**, for example:

   `set http_proxy=http://127.0.0.1:33210
   set https_proxy=http://127.0.0.1:33210`

4. **To utilize the HMove tool, you can invoke the following command to perform a single refactoring prediction in the command line window:**

   `python new_main.py HMove --move_file your\path\to\source\java\file --start_line number --moveto_file your\path\to\target\java\file`

   Note: After  `--start_line` , you need to fill in the number of starting lines of the method to be detected in the source file.

   Example:

   ​	`python new_main.py HMove --move_file E:\cam\WCWidth.java --start_line 13 --moveto_file D:\ConsoleReader.java`

5. Intermediate files generated during program execution will be stored in the folder named starting with "`output`" under the directory `hmove_tool` for viewing.

   

### Usage of FeTruth tool

1. First, download the integrated `fetruth` project from the website https://anonymous.4open.science/r/feTruth-F16E, and place this complete packaged project in the `hmove_tool` folder. Next, open the command line and navigate to the directory containing `new_main.py`.

2. To utilize the FeTruth tool, you can use the following command-line format to detect methods exhibiting feature envy refactoring prediction.

   `python new_main.py FeTruth your\path\to\the\project\under\examination.`

   Example:

   `python new_main.py FeTruth C:\Users\superlit77\Desktop\jsoup`



## Use examples

### Use examples of HMove

You can determine whether the method being detected needs to be moved by checking the output of the program on the command line.

If the program finally outputs: 

> ​	We recommend to move the method starting at line 13 in the source class to the target class. 

Explain that our tool suggests that the method be moved and outputs the target class information.

if the program finally outputs: 

> ​	The method starting at line 13 in the source class is not recommended to move to the target class.

This means that our tool doesn't recommend moving the method to the target class.



A specific example of how to use it is shown below：

​	In this example, we want to check if a method in the source file starting from line 13 needs to be moved to the target file, where the absolute path of the source java file is: `E:camWCWidth.java` and the absolute path of the object file is: `D:ConsoleReader.java`. 

​	Note that the intermediate files output by the program are stored in the `E:hmovesrcoutput_2024-05-14_16-02-26` directory, so that the user can view and understand the detection process at any time.

```
(hmove) E:\hmove\src>python new_main.py HMove --move_file E:\cam\WCWidth.java --start_line 13 --moveto_file D:\ConsoleReader.java
Intermediate files will be stored in E:\hmove\src\output_2024-05-14_16-02-26
----------------------------------------------------------------------------
Start To Load Pretrain Model ... ...
Finish Loading Pretrain Model ! !
----------------------------------------------------------------------------
The operation is complete. All java files have been copied successfully.
Subprocess output: Parent path as string: output_2024-05-14_16-02-26
File created successfully: output_2024-05-14_16-02-26\relation.csv
... initializing  environment ...
Find Java File: E:\hmove\src\output_2024-05-14_16-02-26\source\ConsoleReader.java
Find Java File: E:\hmove\src\output_2024-05-14_16-02-26\source\WCWidth.java
... collecting  relations ...
119
Finish!

Subprocess finished successfully
Subprocess completed, continuing main program execution.
----------------------------------------------------------------------------
We recommend to move the method starting at line 13 in the source class to the target class.

```



### Use examples of FeTruth

FeTruth is designed to detect feature envy within a Java project. It takes the address of the project to be tested as input and outputs detected instances in the format 

​										"`Refactoring Type    Source Method     Target Class`", 

which are listed in the command-line window.

If there are no entries recorded in the columns Refactoring Type, Source Method, and Target Class, it indicates that no methods with feature envy were detected in the project.



A command-line window using the FeTruth tool is documented below as an example.

```
(hmove) E:\HMove\phmove\phmove\tool\hmove_tool>python new_main.py FeTruth C:\Users\superlit77\Desktop\jsoup
E:\HMove\phmove\phmove\tool\hmove_tool\feTruth\dist\feTruth\feTruth.exe
Executing command: E:\HMove\phmove\phmove\tool\hmove_tool\feTruth\dist\feTruth\feTruth.exe -a C:\Users\superlit77\Desktop\jsoup
Preprocessing testing data...
Detecting feature envy methods...
WARNING:tensorflow:AutoGraph could not transform <function Model.make_predict_function.<locals>.predict_function at 0x000001C82AC6D840> and will run it as-is.
Please report this to the TensorFlow team. When filing the bug, set the verbosity to 10 (on Linux, `export AUTOGRAPH_VERBOSITY=10`) and attach the full output.
Cause: Unable to locate the source code of <function Model.make_predict_function.<locals>.predict_function at 0x000001C82AC6D840>. Note that functions defined in certain environments, like the interactive Python shell do not expose th
eir source code. If that is the case, you should to define them in a .py source file. If you are certain the code is graph-compatible, wrap the call using @tf.autograph.do_not_convert. Original error: could not get source code
To silence this warning, decorate the function with @tf.autograph.experimental.do_not_convert
Refactoring Type        Source Method   Target Class
Move Method     org.jsoup.helper.CookieUtil::applyCookiesToRequest(org.jsoup.helper.HttpConnection.Request, java.net.HttpURLConnection):void     org.jsoup.internal.StringUtil
Move Method     org.jsoup.helper.DataUtil::openStream(java.nio.file.Path):java.io.InputStream    org.jsoup.internal.Normalizer
Move Method     org.jsoup.helper.DataUtil::mimeBoundary():java.lang.String       org.jsoup.internal.StringUtil
Move Method     org.jsoup.helper.HttpConnection.Response::createConnection(org.jsoup.helper.HttpConnection.Request):java.net.HttpURLConnection   org.jsoup.helper.CookieUtil
Move Method     org.jsoup.helper.W3CDom::asString(org.w3c.dom.Document, java.util.Map<java.lang.String,java.lang.String>):java.lang.String       org.jsoup.internal.StringUtil
Move Method     org.jsoup.helper.W3CDom::convert(org.jsoup.nodes.Element, org.w3c.dom.Document):void     org.jsoup.helper.W3CDom.W3CBuilder
Move Method     org.jsoup.Jsoup::parse(java.io.File, java.lang.String, java.lang.String):org.jsoup.nodes.Document        org.jsoup.helper.DataUtil
Move Method     org.jsoup.Jsoup::parse(java.io.File, java.lang.String):org.jsoup.nodes.Document          org.jsoup.helper.DataUtil
Move Method     org.jsoup.Jsoup::parse(java.io.File):org.jsoup.nodes.Document    org.jsoup.helper.DataUtil
Move Method     org.jsoup.Jsoup::parse(java.io.File, java.lang.String, java.lang.String, org.jsoup.parser.Parser):org.jsoup.nodes.Document       org.jsoup.helper.DataUtil
Move Method     org.jsoup.Jsoup::parse(java.nio.file.Path, java.lang.String, java.lang.String):org.jsoup.nodes.Document          org.jsoup.helper.DataUtil
Move Method     org.jsoup.Jsoup::parse(java.nio.file.Path, java.lang.String):org.jsoup.nodes.Document    org.jsoup.helper.DataUtil
Move Method     org.jsoup.Jsoup::parse(java.nio.file.Path):org.jsoup.nodes.Document      org.jsoup.helper.DataUtil
Move Method     org.jsoup.Jsoup::parse(java.nio.file.Path, java.lang.String, java.lang.String, org.jsoup.parser.Parser):org.jsoup.nodes.Document         org.jsoup.helper.DataUtil
Move Method     org.jsoup.Jsoup::parse(java.io.InputStream, java.lang.String, java.lang.String):org.jsoup.nodes.Document         org.jsoup.helper.DataUtil
Move Method     org.jsoup.Jsoup::parse(java.io.InputStream, java.lang.String, java.lang.String, org.jsoup.parser.Parser):org.jsoup.nodes.Document        org.jsoup.helper.DataUtil
Move Method     org.jsoup.Jsoup::parseBodyFragment(java.lang.String):org.jsoup.nodes.Document    org.jsoup.parser.Parser
Move Method     org.jsoup.nodes.Element::appendNormalisedText(java.lang.StringBuilder, org.jsoup.nodes.TextNode):void    org.jsoup.internal.StringUtil
Move Method     org.jsoup.nodes.Element::isFormatAsBlock(org.jsoup.nodes.Document.OutputSettings):boolean        org.jsoup.nodes.Document.OutputSettings
Move Method     org.jsoup.nodes.Entities::escape(java.lang.String, org.jsoup.nodes.Document.OutputSettings):java.lang.String     org.jsoup.internal.StringUtil
Move Method     org.jsoup.nodes.Entities::escape(java.lang.Appendable, java.lang.String, org.jsoup.nodes.Document.OutputSettings, boolean, boolean, boolean, boolean):void       org.jsoup.internal.StringUtil
Move Method     org.jsoup.parser.HtmlTreeBuilder::isMathmlTextIntegration(org.jsoup.nodes.Element):boolean       org.jsoup.internal.StringUtil
Move Method     org.jsoup.parser.HtmlTreeBuilder::isHtmlIntegration(org.jsoup.nodes.Element):boolean     org.jsoup.internal.Normalizer
Move Method     org.jsoup.parser.HtmlTreeBuilder::insertElementFor(org.jsoup.parser.Token.StartTag):org.jsoup.nodes.Element      org.jsoup.parser.Tag
Move Method     org.jsoup.parser.HtmlTreeBuilder::isSpecial(org.jsoup.nodes.Element):boolean     org.jsoup.internal.StringUtil
Move Method     org.jsoup.parser.HtmlTreeBuilderState::isWhitespace(org.jsoup.parser.Token):boolean      org.jsoup.parser.Token.Character
Move Method     org.jsoup.parser.HtmlTreeBuilderState::handleRcData(org.jsoup.parser.Token.StartTag, org.jsoup.parser.HtmlTreeBuilder):void      org.jsoup.parser.TreeBuilder
Move Method     org.jsoup.parser.Parser::parseBodyFragment(java.lang.String, java.lang.String):org.jsoup.nodes.Document          org.jsoup.nodes.Document
Move Method     org.jsoup.parser.TokeniserState::readEndTag(org.jsoup.parser.Tokeniser, org.jsoup.parser.CharacterReader, org.jsoup.parser.TokeniserState, org.jsoup.parser.TokeniserState):void         org.jsoup.parser.CharacterReader
Move Method     org.jsoup.parser.TokenQueue::unescape(java.lang.String):java.lang.String         org.jsoup.internal.StringUtil
Move Method     org.jsoup.parser.TokenQueue::escapeCssIdentifier(java.lang.String):java.lang.String      org.jsoup.internal.StringUtil
Move Method     org.jsoup.parser.TreeBuilder::trackNodePosition(org.jsoup.nodes.Node, boolean):void      org.jsoup.parser.Token

The feature envy methods are appended to the output folder.
feTruth.exe executed successfully.
```

