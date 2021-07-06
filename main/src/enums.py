from enum import Enum


class EnumGitCheck(Enum,str):
    GITCHECK = "gitcheck"
    NOGITCHECK = "nogitcheck"