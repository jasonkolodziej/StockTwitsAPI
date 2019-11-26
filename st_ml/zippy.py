import zipfile
import os, sys
zip_folder = 'data.zip'
def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

def unzipdir(ziph, path = ''):
        # Create a ZipFile Object and load sample.zip in it
    ziph.extractall(path)
    dd = lambda x : 'different' if(x != '') else 'current running'
    print('Extract all files in ZIP to {} directory {}'.format(dd(path),path))

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) > 1:
        for a in args:
            print(a)
        exit(-1)
    elif args[0] == 'zip':
        zipf = zipfile.ZipFile(zip_folder, 'w', zipfile.ZIP_DEFLATED)
        zipdir('data', zipf)
        zipf.close()
    elif args[0] == 'unzip':
        unzipf = zipfile.ZipFile(zip_folder, 'r')
        unzipdir(unzipf)
        unzipf.close()
    else:
        for a in args:
            print(a)
        exit(-2)
