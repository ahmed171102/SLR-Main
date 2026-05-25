%----------------------------------------------------------------------------------------------------
%   AAST, Computer Engineering Department LaTEX template
%   Use this template as a base to help you write your thesis (MSc and Phd) 
%   or your graduation project (BSc) documentation.
%
%   This version is created on the 5th of May, 2023
%   Build this template using LuaLatex engine
% 
%   Class File Version 1.0 (4/5/23)
%
%   The department LaTEX transformation project is currently maintained (as per LPPL v1.3c) by: -
% 
%   Sherif Fadel, PhD
%   Noureldin S. Eissa, PhD
%
%   Copyrights(c) AASTMT, Sheraton, Cairo, Computer Engineering Dept.
%
%   Class license:
%   LPPL v1.3c (http://www.latex-project.org/lppl)
%   All modified components must be identified "clearly and unambiguously", and must be declared as 
%   modified versions, both in the source and also when called in some sort of interactive mode. 
%   This is a requirement of LPPL v1.3c
%
%----------------------------------------------------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}[1996/12/01]
\newcommand{\classname}{AASTCOMPUTER}
\ProvidesClass{\classname}[2017/08/27 v1.6 LaTeXTemplates.com]
\providecommand{\baseclass}{book}
\RequirePackage{etoolbox}
\RequirePackage{xparse}
\newbool{nolistspace}
\newbool{chapteroneline}
\newbool{listtoc}
\newbool{toctoc}
\newbool{parskip}
\newbool{headsepline}
\newbool{consistentlayout}
\newbool{inappendix}
\newcommand{\appendixstart}{\appendix \booltrue{inappendix}}

\DeclareOption{nolistspacing}{\booltrue{nolistspace}}
\DeclareOption{liststotoc}{\booltrue{listtoc}}
\DeclareOption{chapterinoneline}{\booltrue{chapteroneline}}
\DeclareOption{toctotoc}{\booltrue{toctoc}}
\DeclareOption{parskip}{\booltrue{parskip}}
\DeclareOption{headsepline}{\booltrue{headsepline}}
\DeclareOption{consistentlayout}{\booltrue{consistentlayout}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{\baseclass}}

\ProcessOptions\relax
\LoadClass{\baseclass}

\ProvideDocumentCommand{\abovechapterskip}{}{\vspace*{-2cm}}
\ProvideDocumentCommand{\chapterbelowskip}{}{\vspace*{40pt}}
\ProvideDocumentCommand{\chapterinbetweenskip}{}{\vspace*{20pt}}
\ProvideDocumentCommand{\autodot}{}{}
\ProvideDocumentCommand{\mdtChapapp}{}{}
\ProvideDocumentCommand{\chapteralign}{}{\centering}
\ProvideDocumentCommand{\chapterfont}{}{\Large\bfseries}
\ProvideDocumentCommand{\chapterprefixfont}{}{\LARGE\bfseries}
\DeclareDocumentCommand{\@makechapterhead}{ m }{%
	\abovechapterskip
	\ifbool{inappendix}{}{	
	\ifnum\value{chapter}=1
	\begin{center}\textit{\large{C h a p t e r~~O n e }}\\ \end{center}
	\fi
	\ifnum\value{chapter}=2
	\begin{center}\textit{\large{C h a p t e r~~T w o }}\\ \end{center}
	\fi	
	\ifnum\value{chapter}=3
	\begin{center}\textit{\large{C h a p t e r~~T h r e e }}\\ \end{center}
	\fi	
	\ifnum\value{chapter}=4
	\begin{center}\textit{\large{C h a p t e r~~F o u r }}\\ \end{center}
	\fi	
	\ifnum\value{chapter}=5
	\begin{center}\textit{\large{C h a p t e r~~F i v e }}\\ \end{center}
	\fi	
	\ifnum\value{chapter}=6
	\begin{center}\textit{\large{C h a p t e r~~S i x }}\\ \end{center}
	\fi	
	\ifnum\value{chapter}=7
	\begin{center}\textit{\large{C h a p t e r~~S e v e n }}\\ \end{center}
	\fi	
	\ifnum\value{chapter}=8
	\begin{center}\textit{\large{C h a p t e r~~E i g h t }}\\ \end{center}
	\fi	
	\ifnum\value{chapter}=9
	\begin{center}\textit{\large{C h a p t e r~~N i n e }}\\ \end{center}
	\fi	
	\ifnum\value{chapter}=10
	\begin{center}\textit{\large{C h a p t e r~~T e n }}\\ \end{center}
	\fi	
	\ifnum\value{chapter}=11
	\begin{center}\textit{\large{C h a p t e r~~E l e v e n }}\\ \end{center}
	\fi	
	\ifnum\value{chapter}=12
	\begin{center}\textit{\large{C h a p t e r~~T w e l v e }}\\ \end{center}
	\fi	
		}
	{\parindent \z@ \chapteralign \normalfont
		\ifnum \c@secnumdepth >\m@ne
		\if@mainmatter
		\ifbool{chapteroneline}{%
			\Large \textbf{\mdtChapapp\thechapter\autodot\enspace}
		}{%
			\chapterprefixfont \@chapapp\space \MakeUppercase{\thechapter}
			\par\nobreak
			\chapterinbetweenskip
		}%
		\fi
		\fi
		\interlinepenalty\@M%\\
		\Large \textbf{\uppercase{#1}}\par\nobreak
		\chapterbelowskip
	}
	\thispagestyle{\chapter@p@gestyle}
}
\def\@makeschapterhead#1{%
	\abovechapterskip
	{\parindent \z@ \chapteralign
		\normalfont
		\interlinepenalty\@M
		\chapterfont  #1\par\nobreak
		\chapterbelowskip
	}
	\thispagestyle{\chapter@p@gestyle}
}

\ProvideDocumentCommand{\addchap}{ s o m }{
	\chapter*{#3}
	\markboth{}{}
	\IfBooleanTF{#1}{
	}{%
		\IfNoValueTF{#2}{
			\addchaptertocentry{#3}
			\markboth{\MakeMarkcase{#3}}{\MakeMarkcase{#3}}
		}{
			\addchaptertocentry{#2}
			\markboth{\MakeMarkcase{#2}}{\MakeMarkcase{#2}}
		}%
	}%
}%

\ProvideDocumentCommand{\addsec}{ s o m }{%
	\section*{#3}%
	\markright{}%
	\IfBooleanTF{#1}{%
	}{%
		\IfNoValueTF{#2}{%
			\addcontentsline{toc}{section}{#3}%
			\markright{\MakeMarkcase{#3}}%%
		}{%
			\addcontentsline{toc}{section}{#2}%
			\markright{\MakeMarkcase{#2}}%
		}%
	}%
}%

\ifbool{parskip}{\RequirePackage{parskip}} 

\ifbool{listtoc}{
	\patchcmd{\listoftables}{\@starttoc{lot}}{
		\addchaptertocentry{List of Tables}\@starttoc{lot}
	}{}{}
	\patchcmd{\listoffigures}{\@starttoc{lof}}{
		\addchaptertocentry{List of Figures}\@starttoc{lof}
	}{}{}
 
}

\ifbool{toctoc}{
	\patchcmd{\tableofcontents}{\@starttoc{toc}
}{
	\addchaptertocentry{\contentsname}\@starttoc{toc}}{}{}
}

\patchcmd{\tableofcontents}{\MakeUppercase}{\MakeMarkcase}{}{}
\patchcmd{\tableofcontents}{\MakeUppercase}{\MakeMarkcase}{}{}
\patchcmd{\listoffigures}{\MakeUppercase}{\MakeMarkcase}{}{}
\patchcmd{\listoffigures}{\MakeUppercase}{\MakeMarkcase}{}{}
\patchcmd{\listoftables}{\MakeUppercase}{\MakeMarkcase}{}{}
\patchcmd{\listoftables}{\MakeUppercase}{\MakeMarkcase}{}{}

\ifbool{nolistspace}{
	\patchcmd{\listoffigures}{%
		\@starttoc{lof}
	}{%
		\begingroup%
		\singlespace\@starttoc{lof}\endgroup%
	}{}{}%
	\patchcmd{\listoftables}{%
		\@starttoc{lot}
	}{%
		\begingroup%
		\singlespace\@starttoc{lot}\endgroup%
	}{}{}%
	\patchcmd{\tableofcontents}{%
		\@starttoc{toc}
	}{%
		\begingroup%
		\singlespace\@starttoc{toc}\endgroup%
	}{}{}%
}{}

\newcommand\fonteighteen{\@setfontsize\fonteighteen{20pt}{24pt}\selectfont}

\RequirePackage{scrbase}
\RequirePackage{scrhack}
\RequirePackage{setspace}
\RequirePackage{longtable} 
\RequirePackage{siunitx} 
\RequirePackage{graphicx} 
\graphicspath{{Figures/}{./}} 
\RequirePackage{booktabs} 
\RequirePackage{caption} 
\RequirePackage{datetime}
\RequirePackage{changepage}
\RequirePackage[normalem]{ulem}
\RequirePackage{xpatch}
\RequirePackage{lipsum}
\RequirePackage[explicit]{titlesec}

\titleformat{\section}
  {\normalfont\large\bfseries}{\thesection \quad \MakeUppercase{#1}}{1em}{}
\titleformat{\subsection}
  {\normalfont\large\bfseries}{\thesubsection \quad #1}{1em}{}
  
\captionsetup{,font=small}
\captionsetup[figure]{labelfont=bf, textfont=bf, name={Figure},labelsep=colon,skip=6pt, belowskip=24pt,font=footnotesize, justification=centerlast}
\captionsetup[table]{labelfont=bf, textfont=bf, name={Table},labelsep=colon,skip=12pt, belowskip=12pt,font=normalsize, justification=raggedright, singlelinecheck=false}

\newcolumntype{P}[1]{>{\centering\arraybackslash}m{#1}} 

\newdateformat{monthyeardate}{\monthname[\THEMONTH] \THEYEAR}

\usepackage[style=american]{csquotes}
\usepackage{tocloft}
\usepackage[backend=bibtex,style=numeric-comp, url=false, natbib=true, maxnames=1, sorting=none]{biblatex}
\DeclareFieldFormat[article]{volume}{, \bibstring{jourvol}\addnbspace #1}
\DeclareFieldFormat[article]{number}{, \bibstring{number}\addnbspace #1}
\DefineBibliographyExtras{english}{%
  \uspunctuation
}
\DeclareFieldFormat{biblabeldate}{#1}

\renewbibmacro{in:}{%
  \ifentrytype{incollection}{\printtext{in }} {\ifentrytype{inproceedings}{\printtext{in \textit{Proc.} }}{}}}


\renewcommand*{\newunitpunct}{\addcomma\space}

\renewbibmacro*{issue+date}{}       
\renewbibmacro*{note+pages}{        
  \printfield{note}
  \setunit{\bibpagespunct}
  \printfield{pages}
  \setunit{\addcomma\addspace}
  \usebibmacro{date}                
  \newunit}

\xpatchbibmacro{volume+number+eid}{
  \setunit*{\adddot}%
}{
}{}{}
 
\xpatchbibdriver{article}
  {\usebibmacro{title}
   \newunit}
  {\usebibmacro{title}%
   \printunit{\addcomma\space}}
  {}
  {}

\renewcommand{\cftchapdotsep}{\cftdotsep}
\setlength{\cftbeforetoctitleskip}{-3em}
\setlength{\cftbeforelottitleskip}{-3em}
\setlength{\cftbeforeloftitleskip}{-3em}

 \renewcommand{\listfigurename}%
     {\begin{center}\Large \MakeUppercase{List of Figures} \end{center}}
\renewcommand{\listtablename}%
     {\begin{center}\Large \MakeUppercase{List of tables} \end{center}}
\renewcommand{\contentsname}%
     {\begin{center}\Large \MakeUppercase{Table of contents} \end{center}}
\renewcommand{\bibname}%
     {\begin{center}\Large \MakeUppercase{References} \end{center}}

\NewDocumentCommand{\thesistitle} { o m }{%
 \IfValueTF{#1}{\def\shorttitle{#1}}{\def\shorttitle{#2}}%
 \def\@title{#2}%
 \def\ttitle{#2}%
}

\DeclareDocumentCommand{\arabicthesistitle}{m}{\newcommand{\arabicthesistitlename}{#1}}
\DeclareDocumentCommand{\author}{m}{\newcommand{\authorname}{#1}\renewcommand{\@author}{#1}}
\DeclareDocumentCommand{\arabicauthor}{m}{\newcommand{\arabicauthorname}{#1}}
\DeclareDocumentCommand{\authortwo}{m}{\newcommand{\authortwoname}{#1}}
\DeclareDocumentCommand{\arabicauthortwo}{m}{\newcommand{\arabicauthortwoname}{#1}}
\DeclareDocumentCommand{\authorthree}{m}{\newcommand{\authorthreename}{#1}}
\DeclareDocumentCommand{\arabicauthorthree}{m}{\newcommand{\arabicauthorthreename}{#1}}
\DeclareDocumentCommand{\authorfour}{m}{\newcommand{\authorfourname}{#1}}
\DeclareDocumentCommand{\arabicauthorfour}{m}{\newcommand{\arabicauthorfourname}{#1}}
\DeclareDocumentCommand{\authorfive}{m}{\newcommand{\authorfivename}{#1}}
\DeclareDocumentCommand{\arabicauthorfive}{m}{\newcommand{\arabicauthorfivename}{#1}}
\DeclareDocumentCommand{\authorsix}{m}{\newcommand{\authorsixname}{#1}}
\DeclareDocumentCommand{\arabicauthorsix}{m}{\newcommand{\arabicauthorsixname}{#1}}
\NewDocumentCommand{\supervisor}{m}{\newcommand{\supname}{#1}}
\NewDocumentCommand{\supervisorspeciality}{m}{\newcommand{\supspec}{#1}}
\NewDocumentCommand{\supervisoruniversity}{m}{\newcommand{\supuni}{#1}}
\NewDocumentCommand{\supervisortwo}{m}{\newcommand{\suptwoname}{#1}}
\NewDocumentCommand{\supervisortwospeciality}{m}{\newcommand{\suptwospec}{#1}}
\NewDocumentCommand{\supervisortwouniversity}{m}{\newcommand{\suptwouni}{#1}}
\NewDocumentCommand{\supervisorthree}{m}{\newcommand{\supthreename}{#1}}
\NewDocumentCommand{\supervisorthreespeciality}{m}{\newcommand{\supthreespec}{#1}}
\NewDocumentCommand{\supervisorthreeuniversity}{m}{\newcommand{\supthreeuni}{#1}}
\NewDocumentCommand{\numberofexaminers}{m}{\newcommand{\numexaminers}{#1}}
\NewDocumentCommand{\numberofstudents}{m}{\newcommand{\numstudents}{#1}}
\NewDocumentCommand{\examiner}{m}{\newcommand{\examname}{#1}}
\NewDocumentCommand{\examineruniversity}{m}{\newcommand{\examuni}{#1}}
\NewDocumentCommand{\examinertwo}{m}{\newcommand{\examtwoname}{#1}}
\NewDocumentCommand{\examinertwouniversity}{m}{\newcommand{\examtwouni}{#1}}
\NewDocumentCommand{\examinerthree}{m}{\newcommand{\examthreename}{#1}}
\NewDocumentCommand{\examinerthreeuniversity}{m}{\newcommand{\examthreeuni}{#1}}
\NewDocumentCommand{\arabicsupervisor}{m}{\newcommand{\arabicsupname}{#1}}
\NewDocumentCommand{\arabicsupervisorspeciality}{m}{\newcommand{\arabicsupspec}{#1}}
\NewDocumentCommand{\arabicsupervisoruniversity}{m}{\newcommand{\arabicsupuni}{#1}}
\NewDocumentCommand{\arabicsupervisortwo}{m}{\newcommand{\arabicsuptwoname}{#1}}
\NewDocumentCommand{\arabicsupervisortwospeciality}{m}{\newcommand{\arabicsuptwospec}{#1}}
\NewDocumentCommand{\arabicsupervisortwouniversity}{m}{\newcommand{\arabicsuptwouni}{#1}}
\NewDocumentCommand{\arabicsupervisorthree}{m}{\newcommand{\arabicsupthreename}{#1}}
\NewDocumentCommand{\arabicsupervisorthreespeciality}{m}{\newcommand{\arabicsupthreespec}{#1}}
\NewDocumentCommand{\arabicsupervisorthreeuniversity}{m}{\newcommand{\arabicsupthreeuni}{#1}}

\NewDocumentCommand{\arabicexaminer}{m}{\newcommand{\arabicexamname}{#1}}
\NewDocumentCommand{\arabicexamineruniversity}{m}{\newcommand{\arabicexamuni}{#1}}
\NewDocumentCommand{\arabicexaminertwo}{m}{\newcommand{\arabicexamtwoname}{#1}}
\NewDocumentCommand{\arabicexaminertwouniversity}{m}{\newcommand{\arabicexamtwouni}{#1}}
\NewDocumentCommand{\arabicexaminerthree}{m}{\newcommand{\arabicexamthreename}{#1}}
\NewDocumentCommand{\arabicexaminerthreeuniversity}{m}{\newcommand{\arabicexamthreeuni}{#1}}

\NewDocumentCommand{\degree}{m}{\newcommand{\degreename}{#1}}
\NewDocumentCommand{\numberofsupervisors}{m}{\newcommand{\supernum}{#1}}
\NewDocumentCommand{\addresses}{m}{\newcommand{\addressname}{#1}}
\NewDocumentCommand{\department}{m}{\newcommand{\deptname}{#1}}
\NewDocumentCommand{\group}{m}{\newcommand{\groupname}{#1}}
\NewDocumentCommand{\faculty}{m}{\newcommand{\facname}{#1}}
\NewDocumentCommand{\subject}{m}{\newcommand{\subjectname}{#1}}
\NewDocumentCommand{\keywords}{m}{\newcommand{\keywordnames	}{#1}}

\newcommand{\checktoopen}{ 
	\if@openright\cleardoublepage\else\clearpage\fi
	\ifdef{\phantomsection}{\phantomsection}{}
}

\newcommand{\HRule}{\rule{.9\linewidth}{.6pt}}
\newcommand{\decoRule}{\rule{.8\textwidth}{.4pt}} 

\setcounter{tocdepth}{3} 
\ProvideDocumentCommand{\addchaptertocentry}{ m }{
	\addcontentsline{toc}{chapter}{#1}
}

\RequirePackage{geometry}

\raggedbottom

% Penalties
\doublehyphendemerits=10000  
\brokenpenalty=10000 
\widowpenalty=9999 
\clubpenalty=9999 
\interfootnotelinepenalty=9999 

% Headers, Checkpoint Feb 2023
\RequirePackage[markcase=used]{scrlayer-scrpage}
\providepairofpagestyles{thesisSimple}{%
	\clearpairofpagestyles%
	\automark[chapter]{chapter}
	\cfoot[\pagemark \centering]{\pagemark}
}
\ifoot{}
\ofoot{}
\pagestyle{thesisSimple}
\providepairofpagestyles[thesisSimple]{thesis}{
	\automark*[section]{}%
}
\providepairofpagestyles[thesisSimple]{review}{
	\ofoot[\shorttitle/\authorname]{\shorttitle/\authorname}
	\ifoot[\today]{\today}
}
\pagestyle{thesis}
\ifbool{headsepline}{\KOMAoption{headsepline}{true}}{}

\newcommand{\blank@p@gestyle}{empty}
\newcommand{\chapter@p@gestyle}{plain}
\NewDocumentCommand{\blankpagestyle}{ m }{%
	\ClassWarning{\classname}{\string\blankpagestyle\space is
	obsolete,\MessageBreak use \string\setblankpagestyle \space  instead}\renewcommand{\blank@p@gestyle}{}{#1}
}
\NewDocumentCommand{\setblankpagestyle}{ m }{\renewcommand{\blank@p@gestyle}{#1}}
\NewDocumentCommand{\setchapterpagestyle}{ m }{\renewcommand{\chapter@p@gestyle}{#1}}

\DeclareDocumentCommand\cleardoublepage{}{\clearpage\if@twoside \ifodd\c@page\else
	\hbox{}
	\thispagestyle{\blank@p@gestyle}
	\newpage
	\if@twocolumn\hbox{}\newpage\fi\fi\fi%
}


\newcommand\listofabbre
{
	\newpage
  \phantomsection 
  \addchaptertocentry{List of Abbreviations}
  \begin{center}{\Large\MakeUppercase{\textbf{List of Abbreviations}}\par}\end{center}
   \vspace{20pt}
  \@starttoc{listofabbrev}
}

\newcommand{\addabbre}[2]{
  \addtocontents{listofabbrev}{\protect \makeabbretable{#1}{#2}}
}

\newdimen\@lmabbre 
\@lmabbre=0.6em
\newdimen\@lcabbre 
\@lcabbre=8em
\newdimen\@rcabbre 
\@rcabbre=\textwidth
\newdimen\@scabbre 
\@scabbre=3em
\newdimen\@pcabbre
\@pcabbre=0em
\advance\@rcabbre by -\@lmabbre 
\advance\@rcabbre by -\@lcabbre 
\advance\@rcabbre by -\@scabbre 
\advance\@rcabbre by -\@pcabbre 

\def\makeabbretable#1#2{
\noindent\vspace{-0.3\baselineskip}\begin{minipage}[t]{\@lcabbre}
\hspace{\@lmabbre}#1
\end{minipage}
\begin{minipage}[t]{\@pcabbre}-\end{minipage}
$\hspace{\@scabbre}$
\begin{minipage}[t]{\@rcabbre}#2\end{minipage}
}


\newcommand{\titlep}
{
	\thispagestyle{empty}
	\begin{titlepage}
	\begin{center}
	\begin{doublespacing}
	\begin{figure}
	\vspace*{-2cm}
	\centering
	\includegraphics[width=0.3\textwidth]{"aast.jpg"} 
	\end{figure}
	\ifnum \degreename=1
	\centerline{\textbf{\large{ARAB ACADEMY FOR SCIENCE, TECHNOLOGY AND MARITIME TRANSPORT}}}
	\fi
	\ifnum \degreename=2
	\centerline{\textbf{\large{ARAB ACADEMY FOR SCIENCE, TECHNOLOGY AND MARITIME TRANSPORT}}}
	\fi
	\ifnum \degreename=3
	\textbf{\fonteighteen{Arab Academy for Science, Technology and Maritime Transport}}
	\vspace{24pt}	
	\fi
		
	\ifnum \degreename=1
	\textbf{College of Engineering and Technology}
	\fi
	
	\ifnum \degreename=2
	\textbf{College of Engineering and Technology}
	\fi
	
	\ifnum \degreename=3
	\textbf{\Large{College of Engineering and Technology}}
	\fi
	
	\ifnum \degreename=3
	\textbf{\Large{Computer Engineering Department}}
	
	\vspace{24pt}
	\large{B. Sc. Final Year Project}
	\vspace{24pt}	
	\fi
	
	\begin{singlespace}
	\Large{\MakeUppercase{\textbf{\@title}}}
 	\end{singlespace}

	\ifnum \degreename=1	
	\large{By}
	\fi
	\ifnum \degreename=2	
	\large{By}
	\fi
	\ifnum \degreename=3	
	\vspace{24pt}
	\large{Presented By:}
	\fi
	
	\ifnum \degreename=1		
	\large{\textbf{\MakeUppercase{\authorname}}}
	\fi
	\ifnum \degreename=2		
	\large{\textbf{\MakeUppercase{\authorname}}}
	\fi
	\ifnum \degreename=3		
	\ifnum \numstudents=1

	\large{\textit{\authorname}}
	\fi
	
	\ifnum \numstudents=2
	\begin{table}[!ht]
	\centering
	\centerline{
	\begin{tabular}{P{8.0cm}P{8.0cm}}
	\\\large{\textit{\authorname}}   & \large{\textit{\authortwoname}}
	\end{tabular}
	}
	\end{table}			
	
	\fi	

	\ifnum \numstudents=3
	\begin{table}[!ht]
	\centering
	\centerline{
	\begin{tabular}{P{8.0cm}P{8.0cm}}
	\multicolumn{2}{c}{\large{\textit{\authorname}}} \\
	\\\large{\textit{\authortwoname}}   & \large{\textit{\authorthreename}}
	\end{tabular}
	}
	\end{table}			
	\fi	

	\ifnum \numstudents=4
	\begin{table}[!ht]
	\centering
	\centerline{
	\begin{tabular}{P{8.0cm}P{8.0cm}}
	\large{\textit{\authorname}}   & \large{\textit{\authortwoname}} \\
	\\\large{\textit{\authorthreename}}   & \large{\textit{\authorfourname}}
	\end{tabular}
	}
	\end{table}			
	
	\fi	

	\ifnum \numstudents=5
	\begin{table}[!ht]
	\centering
	\centerline{
	\begin{tabular}{P{8.0cm}P{8.0cm}}
	\multicolumn{2}{c}{\large{\textit{\authorname}}} \\
	\\\large{\textit{\authortwoname}}   & \large{\textit{\authorthreename}} \\
	\\\large{\textit{\authorfourname}}   & \large{\textit{\authorfivename}}
	\end{tabular}
	}
	\end{table}			
	
	\fi	
	
	\ifnum\numstudents=6 
	\begin{table}[!ht]
	\centering
	\centerline{
	\begin{tabular}{P{8.0cm}P{8.0cm}}
	\large{\textit{\authorname}}   & \large{\textit{\authortwoname}} \\
	\\\large{\textit{\authorthreename}}   & \large{\textit{\authorfourname}} \\
	\\\large{\textit{\authorfivename}}   & \large{\textit{\authorsixname}}
	\end{tabular}
	}
	\end{table}			
	\fi	
	
	\fi

\ifnum \degreename=1
\large{A Thesis submitted to the AASTMT in partial fulfillment of the requirements for the award of the degree of}
\fi

\ifnum \degreename=2
\large{A Thesis submitted to the AASTMT in partial fulfillment of the requirements for the award of the degree of}
\fi

\ifnum \degreename=1
\large{\textbf{MASTER of SCIENCE}}
\fi
\ifnum \degreename=2
\large{\textbf{DOCTOR of PHILOSOPHY}}
\fi

\ifnum \degreename=1

\normalsize{\textbf{In}}

\normalsize{\textbf{Computer Engineering}}

\ifnum \supernum>1
\large{\textbf{Supervisors}}
\fi

\ifnum \supernum=1
\large{\textbf{Supervisor}}
\fi

\ifnum \supernum=3

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{5.5cm}P{5.5cm}P{5.5cm}}
\textbf{\supname}   & \textbf{\suptwoname}   & \textbf{\supthreename}   \\
\textbf{\supspec} & \textbf{\suptwospec} & \textbf{\supthreespec} \\
\textbf{\supuni} & \textbf{\suptwouni} & \textbf{\supthreeuni}
\end{tabular}
}
\end{table}

\fi

\ifnum \supernum=2

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{7.75cm}P{7.75cm}}
\textbf{\supname}   & \textbf{\suptwoname}   \\
\textbf{\supspec} & \textbf{\suptwospec}  \\
\textbf{\supuni} & \textbf{\suptwouni} 
\end{tabular}
}
\end{table}

\fi

\ifnum \supernum=1

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{7.75cm}P{7.75cm}}
\textbf{\supname}      \\
\textbf{\supspec}   \\
\textbf{\supuni}  
\end{tabular}
}
\end{table}
\fi
\fi

\ifnum \degreename=2

\normalsize{\textbf{In}}

\normalsize{\textbf{Computer Engineering}}

\ifnum \supernum>1
\large{\textbf{Supervisors}}
\fi

\ifnum \supernum=1
\large{\textbf{Supervisor}}
\fi

\ifnum \supernum=3
\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{5.5cm}P{5.5cm}P{5.5cm}}
\textbf{\supname}   & \textbf{\suptwoname}   & \textbf{\supthreename}   \\
\textbf{\supspec} & \textbf{\suptwospec} & \textbf{\supthreespec} \\
\textbf{\supuni} & \textbf{\suptwouni} & \textbf{\supthreeuni}
\end{tabular}
}
\end{table}

\fi

\ifnum \supernum=2

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{7.75cm}P{7.75cm}}
\textbf{\supname}   & \textbf{\suptwoname}   \\
\textbf{\supspec} & \textbf{\suptwospec}  \\
\textbf{\supuni} & \textbf{\suptwouni} 
\end{tabular}
}
\end{table}

\fi

\ifnum \supernum=1

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{7.75cm}P{7.75cm}}
\textbf{\supname}      \\
\textbf{\supspec}   \\
\textbf{\supuni}  
\end{tabular}
}
\end{table}
\fi
\fi

\ifnum \degreename=3
\vspace{24pt}
\large{Supervised By:}
\ifnum \supernum=3
\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{8.0cm}P{8.0cm}}
\multicolumn{2}{c}{\textit{\large{\supname}}}   \\
\\\textit{\large{\suptwoname}}&\textit{\large{\supthreename}}
\end{tabular}
}
\end{table}

\fi

\ifnum \supernum=2

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{8.0cm}P{8.0cm}}
\textit{\large{\supname}}&\textit{\large{\suptwoname}}
\end{tabular}
}
\end{table}

\fi

\ifnum \supernum=1

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{8.0cm}}
\textit{\large{\supname}}
\end{tabular}
}
\end{table}
\fi

\fi

\vspace*{\fill}
%\monthyeardate
{\scriptsize \monthyeardate\today}\\ % Date
\end{doublespacing}
\end{center}
\end{titlepage}
}


\def\GetAChar#1#2{\directlua{tex.sprint(string.sub("#1",#2,#2))}}

\NewDocumentCommand{\arabictohindi}{m}
{\ifnum #1=1\begin{textarabic}١\end{textarabic}\fi
	\ifnum #1=2\begin{textarabic}٢\end{textarabic}\fi
	\ifnum #1=3\begin{textarabic}٣\end{textarabic}\fi
	\ifnum #1=4\begin{textarabic}٤\end{textarabic}\fi
	\ifnum #1=5\begin{textarabic}٥\end{textarabic}\fi
	\ifnum #1=6\begin{textarabic}٦\end{textarabic}\fi
	\ifnum #1=7\begin{textarabic}٧\end{textarabic}\fi
	\ifnum #1=8\begin{textarabic}٨\end{textarabic}\fi
	\ifnum #1=9\begin{textarabic}٩\end{textarabic}\fi
	\ifnum #1=0\begin{textarabic}٠\end{textarabic}\fi}
\NewDocumentCommand{\inputarabicyear}{m}
{
\arabictohindi{\GetAChar{#1}{1}}\arabictohindi{\GetAChar{#1}{2}}\arabictohindi{\GetAChar{#1}{3}}\arabictohindi{\GetAChar{#1}{4}}	
}
\NewDocumentCommand{\arabicyearmonth}
{}
{
\ifnum\month=1
{\scriptsize \inputarabicyear{\the\year{}}  \begin{Arabic} يناير\end{Arabic}}\\ % Date
\fi
\ifnum\month=2
{\scriptsize \inputarabicyear{\the\year{}}  \begin{Arabic} فبراير\end{Arabic}}\\ % Date
\fi
\ifnum\month=3
{\scriptsize \inputarabicyear{\the\year{}}  \begin{Arabic} مارس\end{Arabic}}\\ % Date
\fi
\ifnum\month=4
{\scriptsize \inputarabicyear{\the\year{}}  \begin{Arabic} إبريل\end{Arabic}}\\ % Date
\fi
\ifnum\month=5
{\scriptsize \inputarabicyear{\the\year{}}  \begin{Arabic} مايو\end{Arabic}}\\ % Date
\fi
\ifnum\month=6
{\scriptsize \inputarabicyear{\the\year{}}  \begin{Arabic} يونيو\end{Arabic}}\\ % Date
\fi
\ifnum\month=7
{\scriptsize \inputarabicyear{\the\year{}}  \begin{Arabic} يوليو\end{Arabic}}\\ % Date
\fi
\ifnum\month=8
{\scriptsize \inputarabicyear{\the\year{}}  \begin{Arabic} أغسطس\end{Arabic}}\\ % Date
\fi
\ifnum\month=9
{\scriptsize \inputarabicyear{\the\year{}}  \begin{Arabic} سبتمبر\end{Arabic}}\\ % Date
\fi
\ifnum\month=10
{\scriptsize \inputarabicyear{\the\year{}}  \begin{Arabic} أكتوبر\end{Arabic}}\\ % Date
\fi
\ifnum\month=11
{\scriptsize \inputarabicyear{\the\year{}}  \begin{Arabic} نوفمبر\end{Arabic}}\\ % Date
\fi
\ifnum\month=12
{\scriptsize \inputarabicyear{\the\year{}}  \begin{Arabic} ديسمبر\end{Arabic}}\\ % Date
\fi
}

\newcommand{\arabictitlep}
{
	\thispagestyle{empty}
	\begin{titlepage}
	\begin{center}
	\begin{doublespacing}
	\begin{figure}
	\vspace*{-2cm}
	\centering
	\includegraphics[width=0.3\textwidth]{"aast.jpg"} 
	\end{figure}
	
	\centerline{\textbf{\large{\begin{Arabic}الأكاديمية العربية للعلوم و التكنولوجيا و النقل البحرى \end{Arabic}}}}
	
	\textbf{\begin{Arabic}كلية الهندسة و التكنولوجيا -  القاهرة\end{Arabic}}
	
	\begin{singlespace}
	\Large{\textbf{\arabicthesistitlename}}
 	\end{singlespace}

	\large{\begin{Arabic}إعداد\end{Arabic}}

	\large{\textbf{\MakeUppercase{\begin{Arabic}\arabicauthorname\end{Arabic}}}}
	
\normalsize{\begin{Arabic} رسالة مقدمة للأكاديمية العربية للعلوم والتكنولوجيا والنقل البحري لإستكمال متطلبات نيل  
\end{Arabic}}

\ifnum \degreename=1
\large{\textbf{\begin{Arabic}الماجستير\end{Arabic}}}
\fi
\ifnum \degreename=2
\large{\textbf{\begin{Arabic}الدكتوراه\end{Arabic}}}
\fi

\normalsize{\begin{Arabic}في\end{Arabic}}

\normalsize{\textbf{\begin{Arabic}هندسة الحاسب الآلي\end{Arabic}}}

\ifnum \supernum>1
\large{\textbf{\begin{Arabic}المشرفين\end{Arabic}}}
\fi

\ifnum \supernum=1
\large{\textbf{\begin{Arabic}المشرف\end{Arabic}}}
\fi


\ifnum \supernum=3

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{5.5cm}P{5.5cm}P{5.5cm}}
\textbf{\begin{Arabic}\arabicsupthreename\end{Arabic}}   & \textbf{\begin{Arabic}\arabicsuptwoname\end{Arabic}}   & \textbf{\begin{Arabic}\arabicsupname\end{Arabic}}   \\
\textbf{\begin{Arabic}\arabicsupthreespec\end{Arabic}} & \textbf{\begin{Arabic}\arabicsuptwospec\end{Arabic}} & \textbf{\begin{Arabic}\arabicsupspec\end{Arabic}} \\
\textbf{\begin{Arabic}\arabicsupthreeuni\end{Arabic}} & \textbf{\begin{Arabic}\arabicsuptwouni\end{Arabic}} & \textbf{\begin{Arabic}\arabicsupuni\end{Arabic}}
\end{tabular}
}
\end{table}

\fi

\ifnum \supernum=2

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{7.75cm}P{7.75cm}}
\textbf{\begin{Arabic}\arabicsuptwoname\end{Arabic}}   & \textbf{\begin{Arabic}\arabicsupname\end{Arabic}}   \\
\textbf{\begin{Arabic}\arabicsuptwospec\end{Arabic}} & \textbf{\begin{Arabic}\arabicsupspec\end{Arabic}}  \\
\textbf{\begin{Arabic}\arabicsuptwouni\end{Arabic}} & \textbf{\begin{Arabic}\arabicsupuni\end{Arabic}} 
\end{tabular}
}
\end{table}

\fi

\ifnum \supernum=1

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{7.75cm}P{7.75cm}}
\textbf{\begin{Arabic}\arabicsupname\end{Arabic}}      \\
\textbf{\begin{Arabic}\arabicsupspec\end{Arabic}}   \\
\textbf{\begin{Arabic}\arabicsupuni\end{Arabic}}  
\end{tabular}
}
\end{table}
\fi


\vspace*{\fill}

\arabicyearmonth

\end{doublespacing}
\end{center}
\end{titlepage}
}

\newcommand{\arabicexaminersp}
{
	\thispagestyle{empty}
	\begin{titlepage}
	\begin{center}
	\begin{doublespacing}
	\begin{figure}
	\vspace*{-2cm}
	\centering
	\includegraphics[width=0.3\textwidth]{"aast.jpg"} 
	\end{figure}
	
	\centerline{\textbf{\large{\begin{Arabic}الأكاديمية العربية للعلوم و التكنولوجيا و النقل البحرى \end{Arabic}}}}
	
	\textbf{\begin{Arabic}كلية الهندسة و التكنولوجيا -  القاهرة\end{Arabic}}
	
	\begin{singlespace}
	\Large{\textbf{\arabicthesistitlename}}
 	\end{singlespace}

	\large{\begin{Arabic}إعداد\end{Arabic}}

	\large{\textbf{\MakeUppercase{\begin{Arabic}\arabicauthorname\end{Arabic}}}}
	
\normalsize{\begin{Arabic} رسالة مقدمة للأكاديمية العربية للعلوم والتكنولوجيا والنقل البحري لإستكمال متطلبات نيل  
\end{Arabic}}

\ifnum \degreename=1
\large{\textbf{\begin{Arabic}الماجستير\end{Arabic}}}
\fi
\ifnum \degreename=2
\large{\textbf{\begin{Arabic}الدكتوراه\end{Arabic}}}
\fi

\normalsize{\begin{Arabic}في\end{Arabic}}

\normalsize{\textbf{\begin{Arabic}هندسة الحاسب الآلي\end{Arabic}}}

\ifnum \supernum=3

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{5.5cm}P{5.5cm}P{5.5cm}}
\textbf{\begin{Arabic}\arabicsupthreename\end{Arabic}}   & \textbf{\begin{Arabic}\arabicsuptwoname\end{Arabic}}   & \textbf{\begin{Arabic}\arabicsupname\end{Arabic}}   \\
\textbf{\begin{Arabic}\arabicsupthreeuni\end{Arabic}} & \textbf{\begin{Arabic}\arabicsuptwouni\end{Arabic}} & \textbf{\begin{Arabic}\arabicsupuni\end{Arabic}} \\
\begin{Arabic})مشرف(\end{Arabic}   & \begin{Arabic})مشرف(\end{Arabic}   & \begin{Arabic})مشرف(\end{Arabic}   \\
\\\hrulefill & \hrulefill & \hrulefill
\end{tabular}
}
\end{table}

\fi

\ifnum \supernum=2

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{5.5cm}P{5.5cm}}
\textbf{\begin{Arabic}\arabicsuptwoname\end{Arabic}}   & \textbf{\begin{Arabic}\arabicsupname\end{Arabic}}   \\
\textbf{\begin{Arabic}\arabicsuptwouni\end{Arabic}} & \textbf{\begin{Arabic}\arabicsupuni\end{Arabic}} \\
\begin{Arabic})مشرف(\end{Arabic}   & \begin{Arabic})مشرف(\end{Arabic}   \\
\\\hrulefill & \hrulefill
\end{tabular}
}
\end{table}

\fi

\ifnum \supernum=1

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{5.5cm}}
\textbf{\begin{Arabic}\arabicsupname\end{Arabic}}   \\
\textbf{\begin{Arabic}\arabicsupuni\end{Arabic}} \\
\begin{Arabic})مشرف(\end{Arabic}   \\
\\\hrulefill
\end{tabular}
}
\end{table}
\fi

\ifnum \numexaminers=3

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{5.5cm}P{5.5cm}P{5.5cm}}
\textbf{\begin{Arabic}\arabicexamthreename\end{Arabic}}   & \textbf{\begin{Arabic}\arabicexamtwoname\end{Arabic}}   & \textbf{\begin{Arabic}\arabicexamname\end{Arabic}}   \\
\textbf{\begin{Arabic}\arabicexamthreeuni\end{Arabic}} & \textbf{\begin{Arabic}\arabicexamtwouni\end{Arabic}} & \textbf{\begin{Arabic}\arabicexamuni\end{Arabic}} \\
\begin{Arabic})ممتحن(\end{Arabic}   & \begin{Arabic})ممتحن(\end{Arabic}   & \begin{Arabic})ممتحن(\end{Arabic}   \\
\\\hrulefill & \hrulefill & \hrulefill
\end{tabular}
}
\end{table}

\fi

\ifnum \numexaminers=2

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{5.5cm}P{5.5cm}}
\textbf{\begin{Arabic}\arabicexamtwoname\end{Arabic}}  & \textbf{\begin{Arabic}\arabicexamname\end{Arabic}}   \\
\textbf{\begin{Arabic}\arabicexamtwouni\end{Arabic}} & \textbf{\begin{Arabic}\arabicexamuni\end{Arabic}} \\
\begin{Arabic})ممتحن(\end{Arabic}  & \begin{Arabic})ممتحن(\end{Arabic}   \\
\\\hrulefill & \hrulefill 
\end{tabular}
}
\end{table}

\fi

\ifnum \numexaminers=1

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{5.5cm}}
\textbf{\begin{Arabic}\arabicexamname\end{Arabic}}   \\
\textbf{\begin{Arabic}\arabicexamuni\end{Arabic}} \\
\begin{Arabic})ممتحن(\end{Arabic}   \\
\\\hrulefill 
\end{tabular}
}
\end{table}
\fi


\vspace*{\fill}

\arabicyearmonth

\end{doublespacing}
\end{center}
\end{titlepage}
}

\newcommand{\examinersp}
{
	\thispagestyle{empty}
	\begin{titlepage}
	\parskip=0pt
	\begin{center}
	\begin{doublespacing}
	\begin{figure}
	\vspace*{-2cm}
	\centering
	\includegraphics[width=0.3\textwidth]{"aast.jpg"} 
	\end{figure}
	
	\centerline{\textbf{\large{ARAB ACADEMY FOR SCIENCE, TECHNOLOGY AND MARITIME TRANSPORT}}}
	
	\textbf{College of Engineering and Technology}
	
	\vspace{20pt}
	%\textbf{Computer Engineering Department}
	
	\begin{singlespace}
	\Large{\MakeUppercase{\textbf{\@title}}}
 	\end{singlespace}

	\large{By}

	\large{\textbf{\MakeUppercase{\authorname}}}
	
\large{A Thesis submitted to the AASTMT in partial fulfillment of the requirements for the award of the degree of}

\ifnum \degreename=1
\large{\textbf{MASTER of SCIENCE}}
\fi
\ifnum \degreename=2
\large{\textbf{DOCTOR of PHILOSOPHY}}
\fi

\normalsize{\textbf{In}}

\normalsize{\textbf{Computer Engineering}}

\ifnum \supernum=3

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{5.5cm}P{5.5cm}P{5.5cm}}
(Supervisor)   & (Supervisor)   & (Supervisor)   \\
\textbf{\supname}   & \textbf{\suptwoname}   & \textbf{\supthreename}   \\
\textbf{\supuni} & \textbf{\suptwouni} & \textbf{\supthreeuni} \\
\\
\hrulefill & \hrulefill & \hrulefill
\end{tabular}
}
\end{table}

\fi

\ifnum \supernum=2

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{7.75cm}P{7.75cm}}
(Supervisor) & (Supervisor) \\
\textbf{\supname}   & \textbf{\suptwoname}   \\
\textbf{\supuni} & \textbf{\suptwouni} \\
\\
\\
\hrule & \hrule
\end{tabular}
}
\end{table}

\fi

\ifnum \supernum=1

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{7.75cm}}
(Supervisor) \\
\textbf{\supname}      \\
\textbf{\supuni}  \\
\\
\\ \hrule
\end{tabular}
}
\end{table}
\fi

\ifnum \numexaminers=3

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{5.5cm}P{5.5cm}P{5.5cm}}
(Examiner) & (Examiner) & (Examiner) \\
\textbf{\examname} & \textbf{\examtwoname} & \textbf{\examthreename} \\
\textbf{\examuni} & \textbf{\examtwouni} & \textbf{\examthreeuni} \\
\\
\hrulefill & \hrulefill & \hrulefill
\end{tabular}
}
\end{table}

\fi

\ifnum \numexaminers=2

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{7.75cm}P{7.75cm}}
(Examiner) & (Examiner) \\
\textbf{\examname} & \textbf{\examtwoname}  \\
\textbf{\examuni} & \textbf{\examtwouni} 
\\
\\ \hrulefill & \hrulefill
\end{tabular}
}
\end{table}

\fi

\ifnum \numexaminers=1

\begin{table}[!ht]
\centering
\centerline{
\begin{tabular}{P{7.75cm}}
(Examiner) \\
\textbf{\examname}   \\
\textbf{\examuni} \\
\\
\hrulefill  
\end{tabular}
}
\end{table}
\fi

\vspace*{\fill}
{\vspace*{-0.5cm} \scriptsize \monthyeardate\today} % Date
 
\end{doublespacing}
\end{center}
\end{titlepage}
}


	\newcommand{\abstracthere}{Abstract}
	\DeclareDocumentEnvironment{abstractp}{}{%
	\cleardoublepage
	\phantomsection	
	\addchaptertocentry{\abstracthere} 
	
		\thispagestyle{plain}
		\begin{center}{\Large\MakeUppercase{\textbf{\abstracthere}}\par}\end{center}
		\vspace*{20pt}
	}
	{
		\vfil\vfil\null
	}


	\newcommand\arabicabstractp[1]
	{%
	\cleardoublepage
	\phantomsection	
	\addchaptertocentry{Arabic Abstract} 
	
		\thispagestyle{plain}
		\begin{center}{\Large\MakeUppercase{\textbf{Arabic Abstract}}\par}\end{center}
		
		\begin{center}{\Large\MakeUppercase{\textbf{\begin{Arabic}الملخص العربي\end{Arabic}}}\par}\end{center}
		
		\vspace*{20pt}
		
		
		#1		
	}
	

\newcommand{\acknowledgmentname}{Acknowledgments}
	\DeclareDocumentEnvironment{acknowledgmentsp}{}{%
	\newpage
	\phantomsection	
	\addchaptertocentry{\acknowledgmentname} 
		\checktoopen
		
		\thispagestyle{plain}
		\begin{center}{\Large\MakeUppercase{\textbf{\acknowledgmentname}}\par}\end{center}
		\vspace*{20pt}
	}
	{
		\vfil\vfil\null
	}


\newcommand{\declarationp}
{	
\thispagestyle{empty}
	
	\begin{titlepage}
	\begin{doublespacing}
	\begin{center}
		\Large{\MakeUppercase{\textbf{\underline{Declaration}}}}
	\end{center}
	\vspace*{\fill}

\ifnum \degreename=1
I \textbf{\authorname} certify that all the material in this thesis that is not my own work has been identified, and that no material is included for which a degree has previously been conferred on me.
\vskip 0.75cm
\noindent The contents of this thesis reflect my own personal views, and are not necessarily endorsed by the University.
\vskip 1cm

\noindent (Signature)\\
\rule[0.5em]{25em}{0.5pt} 
\vskip 0.25cm 
\noindent (Date)~\today\\
\rule[0.5em]{25em}{0.5pt}  
\fi

\ifnum \degreename=2
I \textbf{\authorname} certify that all the material in this thesis that is not my own work has been identified, and that no material is included for which a degree has previously been conferred on me.
\vskip 0.75cm
\noindent The contents of this thesis reflect my own personal views, and are not necessarily endorsed by the University.
\vskip 1cm

\noindent (Signature)\\
\rule[0.5em]{25em}{0.5pt} 
\vskip 0.25cm 
\noindent (Date)~\today\\
\rule[0.5em]{25em}{0.5pt}  
\fi

\ifnum \degreename=3
I hereby certify that this report, which I now submit for assessment on the program of 
study leading to the award of Bachelor of Science in \textit{Computer Engineering}, is all my own work and contains no Plagiarism. By submitting this report, I 
agree to the following terms:

\begin{adjustwidth}{1.0cm}{1.0cm}
\textit{Any text, diagrams or other material copied from other sources (including, but not 
limited to, books, journals, and the internet) have been clearly acknowledged and 
cited followed by the reference number used; either in the text or in a 
footnote/endnote. The details of the used references that are listed at the end of the 
report are confirming to the referencing style dictated by the final year project 
template and are, to my knowledge, accurate and complete.}\end{adjustwidth}

I have read the sections on referencing and plagiarism in the final year project template.\\
\uline{I understand that plagiarism can lead to a reduced or fail grade, in serious cases, for the 
Graduation Project course.}

\vspace{24pt}

	\ifnum \numstudents=1
	\begin{table}[!ht]
	\begin{tabular}{p{8.0cm}}
	Student Name:~\authorname   \\
	\\Registration Number:~\hrulefill   \\
	\\Signed:~\hrulefill \\
	\\Date:~\today 
	\end{tabular}
	\end{table}			
	\fi

	\ifnum \numstudents=2
	\begin{table}[!ht]
	\centering
	\centerline{
	\begin{tabular}{p{8.0cm}p{8.0cm}}
	\textbf{Student Name:}~\authorname   & \textbf{Student Name:}~\authortwoname \\
	\\\textbf{Registration Number:}~\hrulefill   & \textbf{Registration Number:}~\hrulefill \\
	\\\textbf{Signed:}~\hrulefill & \textbf{Signed:}~\hrulefill \\
		
	\end{tabular}
	}
	\end{table}			
	\centerline{\textbf{Date:}~\today} 	
	\fi
	
	
	\ifnum \numstudents=3
	\begin{table}[!ht]
	\centering
	\centerline{
	\begin{tabular}{p{8.0cm}p{8.0cm}}
	\textbf{Student Name:}~\authorname   & \textbf{Student Name:}~\authortwoname \\
	\\\textbf{Registration Number:}~\hrulefill   & \textbf{Registration Number:}~\hrulefill \\
	\\\textbf{Signed:}~\hrulefill & \textbf{Signed:}~\hrulefill \\
	
	&  \\
	\\\textbf{Student Name:}~\authorthreename   & \\
	\\\textbf{Registration Number:}~\hrulefill   & \\
	\\\textbf{Signed:}~\hrulefill & \\
	
	\end{tabular}
	}
	\end{table}			
	\centerline{\textbf{Date:}~\today} 	
	\fi
	

	\ifnum \numstudents=4
	\begin{table}[!ht]
	\centering
	\centerline{
	\begin{tabular}{p{8.0cm}p{8.0cm}}
	\textbf{Student Name:}~\authorname   & \textbf{Student Name:}~\authortwoname \\
	\\\textbf{Registration Number:}~\hrulefill   & \textbf{Registration Number:}~\hrulefill \\
	\\\textbf{Signed:}~\hrulefill & \textbf{Signed:}~\hrulefill \\
	
	&  \\
	\\\textbf{Student Name:}~\authorthreename   & \textbf{Student Name:}~\authorfourname \\
	\\\textbf{Registration Number:}~\hrulefill   & \textbf{Registration Number:}~\hrulefill \\
	\\\textbf{Signed:}~\hrulefill & \textbf{Signed:}~\hrulefill \\
	
	\end{tabular}
	}
	\end{table}			
	\centerline{\textbf{Date:}~\today} 	
	\fi
	

	\ifnum \numstudents=5
	\begin{table}[!ht]
	\centering
	\centerline{
	\begin{tabular}{p{8.0cm}p{8.0cm}}
	\textbf{Student Name:}~\authorname   & \textbf{Student Name:}~\authortwoname \\
	\\\textbf{Registration Number:}~\hrulefill   & \textbf{Registration Number:}~\hrulefill \\
	\\\textbf{Signed:}~\hrulefill & \textbf{Signed:}~\hrulefill \\
	
	&  \\
	\\\textbf{Student Name:}~\authorthreename   & \textbf{Student Name:}~\authorfourname \\
	\\\textbf{Registration Number:}~\hrulefill   & \textbf{Registration Number:}~\hrulefill \\
	\\\textbf{Signed:}~\hrulefill & \textbf{Signed:}~\hrulefill \\
	
	&  \\
	\\\textbf{Student Name:}~\authorfivename   & \\
	\\\textbf{Registration Number:}~\hrulefill   & \\
	\\\textbf{Signed:}~\hrulefill & \\
	
		
	\end{tabular}
	}
	\end{table}			
	\centerline{\textbf{Date:}~\today} 	
	\fi
	

	\ifnum \numstudents=6
	\begin{table}[!ht]
	\centering
	\centerline{
	\begin{tabular}{p{8.0cm}p{8.0cm}}
	\textbf{Student Name:}~\authorname   & \textbf{Student Name:}~\authortwoname \\
	\\\textbf{Registration Number:}~\hrulefill   & \textbf{Registration Number:}~\hrulefill \\
	\\\textbf{Signed:}~\hrulefill & \textbf{Signed:}~\hrulefill \\
	
	&  \\
	\\\textbf{Student Name:}~\authorthreename   & \textbf{Student Name:}~\authorfourname \\
	\\\textbf{Registration Number:}~\hrulefill   & \textbf{Registration Number:}~\hrulefill \\
	\\\textbf{Signed:}~\hrulefill & \textbf{Signed:}~\hrulefill \\
	
	&  \\
	\\\textbf{Student Name:}~\authorfivename   & \textbf{Student Name:}~\authorsixname \\
	\\\textbf{Registration Number:}~\hrulefill   & \textbf{Registration Number:}~\hrulefill \\
	\\\textbf{Signed:}~\hrulefill & \textbf{Signed:}~\hrulefill \\
	
		
	\end{tabular}
	}
	\end{table}			
	\centerline{\textbf{Date:}~\today} 	
	\fi
	\fi
	

	
	\vspace*{\fill}
	\end{doublespacing}
	\end{titlepage}
}


\newcommand\listreferences
{
	\newpage
  	\phantomsection 
  	\addchaptertocentry{References}

  
  \begin{center}{\Large\MakeUppercase{\textbf{References}}\par}\end{center}
 \vspace{20pt}
  \printbibliography[heading=none]

}
\newcommand\dedicationp[1]
{
	\thispagestyle{empty}
	\begin{titlepage}
	\thispagestyle{empty}
	\vspace*{\fill}
	\begin{center}{\Large\slshape #1}\end{center}
	\vspace*{\fill}
	\end{titlepage}
}

	\DeclareDocumentEnvironment{listofpublicationsp}{}{%
	\newpage
	\phantomsection	
	\addchaptertocentry{List of Publications} 
		
		\thispagestyle{plain}
		\begin{center}{\Large\MakeUppercase{\textbf{List of publications}}\par}\end{center}
		\vspace*{20pt}
	}
	{
		\vfil\vfil\null
	}

\AtEndPreamble{\RequirePackage{hyperref}
\hypersetup{pdfpagemode={UseOutlines},
bookmarksopen=true,
bookmarksopenlevel=0,
hypertexnames=false,
colorlinks=false,
citecolor=black,
linkcolor=black,
urlcolor=black,
pdfstartview={FitV},
unicode,
breaklinks=true,
}

\pdfstringdefDisableCommands{
	\let\\\space
}
	}
{
}

\endinput

