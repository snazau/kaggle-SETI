import os
if not os.path.exists('fmix.zip'):
    os.system("wget -O fmix.zip https://github.com/ecs-vlc/fmix/archive/master.zip")
    os.system("unzip -qq fmix.zip")
    os.makedirs("./FMix")
    os.system("mv FMix-master/* FMix/")
    os.system("rm -r FMix-master")
