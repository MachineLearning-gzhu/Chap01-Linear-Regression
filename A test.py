# a test
class student:
    name = "abcd"
    age = 17
    def __init__(self, name, age):
        self.name = name
        self.age = age

a = student("zhang", 15)
print(a.name,a.age)
