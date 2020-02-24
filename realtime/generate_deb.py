from os import system, makedirs

version = input("Version: ")

makedirs(f"wavae_{version}/usr/lib/")
makedirs(f"wavae_{version}/usr/local/lib/pd-externals/")
makedirs(f"wavae_{version}/DEBIAN/")

system(f"cp build/*.pd_linux wavae_{version}/usr/local/lib/pd-externals/")
system(f"cp build/libwavae/libwavae.so wavae_{version}/usr/lib/")

with open(f"wavae_{version}/DEBIAN/control", "w") as control:
    control.write("Package: wavae\n")
    control.write(f"Version: {version}\n")
    control.write("Maintainer: Antoine CAILLON <caillon@ircam.fr>\n")
    control.write("Architecture: all\n")
    control.write(
        "Description: WaVAE puredata external. Needs libtorch in /usr/lib\n")

system(f"dpkg-deb --build wavae_{version}")
system(f"rm -fr wavae_{version}/")