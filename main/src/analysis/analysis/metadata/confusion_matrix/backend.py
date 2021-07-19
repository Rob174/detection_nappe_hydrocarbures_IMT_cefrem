import numpy as np

from bs4 import BeautifulSoup

from main.FolderInfos import FolderInfos

import clipboard


class ConfusionMatrixBackend:
    def __init__(self,dico,access_functions,access_names):
        self.matrix = None
        it = iter(access_functions)
        while self.matrix is None:
            try:
                self.matrix = np.array(eval(next(it))(dico))
            except KeyError:
                pass
            except StopIteration:
                raise Exception(f"Matrix not found with functions {access_functions}")
        self.names = None
        it = iter(access_names)
        while self.names is None:
            try:
                self.names = eval(next(it))(dico)
            except KeyError:
                pass
            except StopIteration:
                raise Exception(f"Matrix not found with functions {access_names}")
        str_list = self.generate_str_list()
        self.add_to_html_web_page(FolderInfos.root_folder+FolderInfos.separator.join(["main","src","analysis","analysis","confusion_matrix","confusion_matrix.html"]),
                                  str_list)
    def generate_str_list(self):
        tot = np.sum(self.matrix[:-1,:-1])
        full_matrix_percent = np.copy(self.matrix) / tot * 100
        num_matrix_classes = len(self.matrix[:-1,:-1])
        final_matrix = np.empty((num_matrix_classes + 1, num_matrix_classes + 1)).tolist()
        for x in range(num_matrix_classes + 1):
            for y in range(num_matrix_classes + 1):
                final_matrix[x][y] = f"{self.matrix[x, y]},{full_matrix_percent[x, y]:.2f}%"
        return final_matrix
    def add_to_html_web_page(self,template_path: str,list_str):
        with open(template_path,"r") as fp:
            file_content = fp.read()
            soup = BeautifulSoup(file_content, 'html.parser')
            # Column names
            for name in (reversed(self.names)):
                tag1 = soup.new_tag("th")
                tag1.string = name
                soup.body.table.thead.select("tr")[1].insert(4,tag1)
            soup.body.table.thead.tr.select("th")[2]["colspan"] = len(self.names)
            # Row names
            soup.body.table.tbody.tr.th["rowspan"] = len(self.names)
            tag1 = soup.new_tag("th")
            tag1.string = self.names[0]
            soup.body.table.tbody.tr.append(tag1)

            def generate_tags(i_l,name):
                lelems = []

                for vals in list_str[i_l][:-1]:
                    td = soup.new_tag("td")
                    vals = vals.split(",")
                    p1 = soup.new_tag("p")
                    p1.string = vals[0]
                    p2 = soup.new_tag("p")
                    p2.string = vals[1]
                    td.append(p1)
                    td.append(p2)
                    lelems.append(td)
                # Adding totals
                th = soup.new_tag("th")
                vals = list_str[i_l][-1]
                vals = vals.split(",")
                p1 = soup.new_tag("p")
                p1.string = vals[0]
                p2 = soup.new_tag("p")
                p2.string = vals[1]
                th.append(p1)
                th.append(p2)
                lelems.append(th)
                return lelems
            tags = generate_tags(0,self.names[0])
            for tag in tags:
                soup.body.table.tbody.tr.append(tag)
            for i,name in zip(range(0,len(list_str)),reversed(self.names[1:])):
                tr = soup.new_tag("tr")
                th = soup.new_tag("th")
                th.string = name
                tr.append(th)
                tags = generate_tags(i,name)
                for tag in tags:
                    tr.append(tag)
                soup.body.table.tbody.insert(2,tr)
            # Totals line
            footer_tr = soup.body.table.tbody.select("tr")[-1]
            for vals in reversed(list_str[-1][:-1]):
                tag = soup.new_tag("th")
                for val in vals.split(","):
                    p = soup.new_tag("p")
                    p.string = val
                    tag.append(p)
                footer_tr.insert(4,tag)
            # Accuracy
            th = soup.new_tag("th")
            ptitle = soup.new_tag("p")
            ptitle.string = "Correct"
            th.append(ptitle)

            val,percent = list_str[-1][-1].split(",")
            pval = soup.new_tag("p")
            pval.string = val
            th.append(pval)

            ppercent = soup.new_tag("p")
            ppercent.string = percent
            th.append(ppercent)

            footer_tr.append(th)
            clipboard.copy(str(soup))
            return str(soup)
