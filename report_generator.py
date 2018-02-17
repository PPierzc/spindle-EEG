# -*- coding: utf-8 -*-
class Page(object):
    def __init__(self, path, title, subtitle, description):
        self.path = path
        self.image_on_left = True
        with open(self.path, 'wb') as f:
            meta = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Report</title>
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <link rel="stylesheet" type="text/css" href="style/style.css">
</head>
<body>
<div class="row">
    <div class="col-md-9">
        <h1>'''+title+'''</h1>
        <h2>'''+subtitle+'''</h2>
        <p>'''+description+'''</p>
    </div>
    <div class="col-md-3">
        <a href="./index.html">Powrót do spisu treści</a>
    </div>
</div><br /><br />'''
            f.write(meta.encode("utf-8"))

    def close(self):
        with open (self.path, 'ab') as f:
            f.write("\n</body>\n</html>".encode("utf-8"))

    def image(self, path, title, description):
        with open (self.path, 'ab') as f:
            if self.image_on_left: align = "left-text"
            else: align = "right-text"

            if self.image_on_left:
                image = '''\n<div class="row item">
    <div class="col-md-6"><img src="'''+path+'''" /></div>
    <div class="col-md-6 description '''+align+'''">
        <h3>'''+title+'''</h3>
        <p class="italic">'''+description+'''</p>
    </div>
</div>'''
            else:
                image = '''\n<div class="row item">
    <div class="col-md-6 description '''+align+'''">
        <h3>''' + title + '''</h3>
        <p class="italic">''' + description + '''</p>
    </div>
    <div class="col-md-6"><img src="''' + path + '''" /></div>
</div>'''
            self.image_on_left = not self.image_on_left
            f.write(image.encode("utf-8"))

    def table(self, left_title, right_title, data, title, description):
        if self.image_on_left:
            align = "left-text"
        else:
            align = "right-text"

        with open(self.path, 'ab') as f:
            f.write('<div class="row item">'.encode("utf-8"))
            # Table
            table = '''\n<div class="col-md-6">
        <div class="row">
            <div class="col-md-2"></div>
            <div class="col-md-4 center table-title">
                '''+left_title+'''
            </div>
            <div class="col-md-4 center table-title">
                '''+right_title+'''
            </div>
            <div class="col-md-2"></div>
        </div>'''
            for i in data:
                if i[0] != data[-1][0] and i[1] != data[-1][1]:
                    table += '''<div class="row">
            <div class="col-md-2"></div>
            <div class="col-md-4 center table-item">
                '''+ str(i[0])+'''
            </div>
            <div class="col-md-4 center table-item">
                '''+ str(i[1]) +'''
            </div>
            <div class="col-md-2"></div>
        </div>'''
                else:
                    table += '''<div class="row">
            <div class="col-md-2"></div>
            <div class="col-md-4 center table-item-bottom">
                ''' + str(i[0]) + '''
            </div>
            <div class="col-md-4 center table-item-bottom">
                ''' + str(i[1]) + '''
            </div>
            <div class="col-md-2"></div>
        </div>'''

            # Description
            content = '''<div class="col-md-6 description '''+align+'''">
            <h3>'''+title+'''</h3>
            <p class="italic">'''+description+'''</p>
        </div>'''
            if self.image_on_left:
                f.write(table.encode("utf-8") + '</div>'.encode("utf-8"))
                f.write(content.encode("utf-8"))
            else:
                f.write(content.encode("utf-8"))
                f.write(table.encode("utf-8") + '</div>'.encode("utf-8"))
            f.write('</div>'.encode("utf-8"))
            self.image_on_left = not self.image_on_left

    def pagination(self, next=None, back=None):
        with open(self.path, 'ab') as f:
            content = '''<div class="footer">'''
            f.write(content.encode("utf-8"))
            if next:
                content = '''<a href="'''+next+'''">
        <div class="btn right">
            Next
        </div>
    </a>'''
                f.write(content.encode("utf-8"))
            if back:
                content = '''<a href="'''+back+'''">
        <div class="btn left">
            Back
        </div>
    </a>'''
                f.write(content.encode("utf-8"))
            f.write("</div>".encode("utf-8"))

    def paragraph(self, content):
        with open(self.path, 'ab') as f:
            f.write("<p>    ".encode("utf-8") + content.encode("utf-8") + "</p>".encode("utf-8"))

    def header(self, header):
        with open(self.path, 'ab') as f:
            f.write("<h4>".encode("utf-8") + header.encode("utf-8") + "</h4>".encode("utf-8"))


if __name__ == '__main__':
    page = Page("./report/test.html", "Test", "Subtitle", "Lorem")
    page.image('./images/signal.png', "Windowed Signal", "The signal is windowed 0 to 20 s")
    page.table('START', 'END', [[20,30], [30,40], [40,50]], "Windowed Signal", "The signal is windowed 0 to 20 s")
    page.image('./images/signal.png', "Windowed Signal", "The signal is windowed 0 to 20 s")
    page.pagination('./1.html', './k.html')
    page.close()