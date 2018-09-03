import pypandoc
import sys
import datetime
currentDate = datetime.datetime.now()

from subprocess import call


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
              '---\n'
    append_to_file(filePath, headers)


def create_markdown_file_from_tex(textFilePath, markdownTargetFilePath):
    pypandoc.convert_file(textFilePath, "markdown", outputfile=markdownTargetFilePath)


# TODO.   Find way of reading whole content to file and parse strings accordingly (or literally call a one liner for perll
def fix_formating_mathjax_equations(filePath):
    regex = '(?<!\$)\$(?!\$)'
    apply_regex_to_file(filePath, regex)


def apply_regex_to_file(filePath, regex):
    call(['perl', '-pi', '-e', 's/(?<!\$)\$(?!\$)/\$\$/g', filePath])


def create_new_post(textFilePath=None, postName=""):
    newPostFilePath = '_posts/{}-{}-{}-{}.{}'.format(currentDate.year, currentDate.month, currentDate.day, postName, 'md')

    create_markdown_file_from_tex(textFilePath, newPostFilePath)

    append_post_metadata_to_post_file(newPostFilePath, title=postName)

    fix_formating_mathjax_equations(newPostFilePath)


def get_current_date():
    date = '{}-{}-{} {}:{}:{}'.format(currentDate.year, currentDate.month, \
                                      currentDate.day, currentDate.hour, \
                                      currentDate.minute, currentDate.second)
    return date


if __name__ == '__main__':
    if not (len(sys.argv) == 3):
        print("TODO: get better help")
        print("First argument should be location of .tex file with post\nSecond argument should be name of post")
    else:
        create_new_post(textFilePath=sys.argv[1], postName=sys.argv[2])
