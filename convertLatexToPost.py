import pypandoc
import sys
from subprocess import call

from docopt import docopt

import datetime
currentDate = datetime.datetime.now()

def get_current_date():
    date = '{}-{}-{} {}:{}:{}'.format(currentDate.year, currentDate.month, \
                                      currentDate.day, currentDate.hour, \
                                      currentDate.minute, currentDate.second)
    return date

def append_to_file(filePath, string):
    with open(filePath, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(string + '\n' + content)


def append_post_metadata_to_post_file(filePath, title=None, tags=[], layout='post'):
    date = '{}-{}-{} {}:{}:{}'.format(currentDate.year, currentDate.month, \
                                      currentDate.day, currentDate.hour, \
                                      currentDate.minute, currentDate.second)
    headers = '---\n' + \
              'layout: {}\n'.format(layout) + \
              'cover: assets/images/shiva.jpg\n' + \
              'title: {}\n'.format(title) + \
              'date: {}\n'.format(date) + \
              'tags: {}\n'.format(tags) + \
              'author: Daniii\n' + \
              '---\n'
    append_to_file(filePath, headers)


def create_markdown_file_from_tex(textFilePath, markdownTargetFilePath, bibliographyPath):
    bibliographyArg = [] # ['--bibliography={}'.format(bibliographyPath)]
    pypandoc.convert_file(textFilePath, "markdown-citations", outputfile=markdownTargetFilePath, extra_args=bibliographyArg)


def fix_formating_mathjax_equations(filePath):
    regex = 's/(?<!\$)\$(?!\$)/\$\$/g'
    apply_regex_to_file(filePath, regex)


def apply_regex_to_file(filePath, regex):
    call(['perl', '-pi', '-e', regex, filePath])


def add_reference_text_to_references(filePath):
    reference_markdown_text = 'Time for References!\n------------\n\n'
    reference_line = "<div id=\"refs\" class=\"references\">\n"
    with open(filePath, "r") as f:
        contents = f.readlines()
    references_index = contents.index(reference_line)
    contents.insert(references_index, reference_markdown_text)
    with open(filePath, "w+") as f:
        f.writelines(contents)


def create_new_post(textFilePath=None, postName="", bibliographyPath=None):
    newPostFilePath = '_posts/{}-{}-{}-{}.{}'.format(currentDate.year, currentDate.month, currentDate.day, postName, 'md')

    create_markdown_file_from_tex(textFilePath, newPostFilePath, bibliographyPath)

    append_post_metadata_to_post_file(newPostFilePath, title=postName)

    fix_formating_mathjax_equations(newPostFilePath)

    # add_reference_text_to_references(newPostFilePath)


if __name__ == '__main__':
    #_USAGE = '''
    # Usage:
    #     convertolatex --source-latex-file=<file> --destination-markdown-file=<file> [--bibtex-file=<file>]
    #     convertolatex --help

    # Options:
    #     --source-latex-file=<file>             Name of the *.tex file containing the source material [default:None]
    #     --destination-markdown-file=<file2>    Name of the new file containing the markdown blogpost [default:None]
    #     --bibtex-file=<file3>                  Bibtex file used to cite references [default:None]
    # '''
    # options = docopt(_USAGE)
    # print("help")
    # if options['--source-latex-file'] is None or options['--destination-markdown-file'] is None:
    #     print("inside if")
    #     print(options['--help'])
    # else: 
    #     print("past if")
    create_new_post(textFilePath=sys.argv[1], postName=sys.argv[2], bibliographyPath=sys.argv[3])
