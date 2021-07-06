from enum import Enum


class EnumGitCheck(str,Enum):
    GITCHECK = "gitcheck"
    NOGITCHECK = "nogitcheck"