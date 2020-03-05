from os import system, makedirs

VERSION = input("Version: ")

makedirs(f"wavae_{VERSION}/usr/lib/")
makedirs(f"wavae_{VERSION}/usr/local/lib/pd-externals/wavae/")
makedirs(f"wavae_{VERSION}/DEBIAN/")

system(
    f"cp build/*.pd_linux wavae_{VERSION}/usr/local/lib/pd-externals/wavae/")
system(f"cp help* wavae_{VERSION}/usr/local/lib/pd-externals/wavae/")

system(f"cp build/libwavae/libwavae.so wavae_{VERSION}/usr/lib/")

with open(f"wavae_{VERSION}/DEBIAN/control", "w") as control:
    control.write("Package: wavae\n")
    control.write(f"Version: {VERSION}\n")
    control.write("Maintainer: Antoine CAILLON <caillon@ircam.fr>\n")
    control.write("Architecture: all\n")
    control.write(
        "Description: WaVAE puredata external. Needs libtorch in /usr/lib\n")

system(f"dpkg-deb --build wavae_{VERSION}")
system(f"rm -fr wavae_{VERSION}/")
