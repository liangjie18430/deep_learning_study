import collections

if __name__ == '__main__':

    a = ["a","b","c"]
    for word,index in enumerate(a):
        print(word,",",index)

    b = [["UNK",-1]]

    c = "This is a test for word count a a b b c c"
    c_wors = c.split()
    d = collections.Counter(c_wors).most_common(5)


    print("before extend b.size:%d]" % (len(b)))
    b.extend(d)

    print("after extend b.size:%d]" % (len(b)))
    for word,index in d:
        print("[word:%s,index:%s]"%(word,index))



    pass