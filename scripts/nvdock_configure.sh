#!/usr/bin/env bash
#
# -------------------------------------------------------------------- #
# Published as part of the TensorFrame project: https://tensorframe.ai.
# Copyright (c) 2018 Anthony Potappel, the Netherlands.
# All Rights Reserved.
#
# This software may be modified and distributed under the terms of the
# MIT license. See the LICENSE file for details.
# -------------------------------------------------------------------- #

DRIVER_VERSION="410.78"
DOCKER_COMPOSE_VERSION="1.21.1"

PROGNAME="nvdock_configure.sh"

function usage(){
    cat << USAGE
  Notes
    $PROGNAME prepares a host to run CUDA based docker containers.
    Current supported distributions: Ubuntu 16.04, Ubuntu 18.04.

  Usage:
    # default install or update
    command:    $PROGNAME [MODE]
    example:    $PROGNAME --server

    # custom install or update
    command:    $PROGNAME --custom [RUNPARTS]
    example:    $PROGNAME --custom docker,nvidia-docker

    # check current configuration
    command:    $PROGNAME --check [CHECKS]
    example:    $PROGNAME --check driver
 
  MODES
    --server            Full install/ Update including driver
    --desktop           Install/ Update excluding driver
    --custom [RUNPARTS] Aply specific CUSTOM options
    --check [CHECKS]    Check for a specific component

  RUNPARTS
    ubuntu              Update and/ or install essential packages
    docker              Install Docker
    nvidia-docker       Install NVIDIA docker scripts
    disable-nouveau     Disable nouveau driver
    driver              Install latest driver
    driver=\${VERSION}  Install NVIDIA \${VERSION}
    docker-restart      Restart docker

  CHECKS
    driver              Verify if NVIDIA driver is loaded
    driver=\${VERSION}  Verify if NVIDIA driver with \${VERSION} is loaded
    runtime             Verify CUDA runtime

USAGE
    return 0
}


function check_fail(){
    return_code="$1"
    fail_msg="$2"

    [[ ! ${return_code} -eq 0 ]] && echo "[FAIL:$fail_msg]"
    return "$return_code"
}


function check_fail_msg(){
    return_code="$1"
    return_msg="$2"
    fail_msg="$3"

    [[ "$return_code" -eq 0 ]] && return 0
    echo -n "${return_msg}: "
    check_fail "$return_code" "$fail_msg"
    return $?
}


function run_as_docker(){
    sg docker "$*"
    return $?
}


function is_group_docker(){
    groups |tr -s " " "\n" |grep -q "^docker$" 2>/dev/null
    return $?
}


function filter_input_argstr(){
    # Accepted input: list of comma separated function names conforming regex: [-a-z0-9=\.].
    # input is lowercased first and then filtered on regex.
    arglist=""
    for arg in "$@";do
        arg_p=$(echo "$arg" \
            |sed 's/\(.*\)/\L\1/g;s/[^-a-z0-9=\.,]//g;s/,,*/,/g;s/^,\|,$//g;s/,/\ /g')
        arglist="${arglist} ${arg_p}"
    done
    echo "${arglist}" |sed 's/^\ //g'
    return 0
}


function p__ubuntu(){
    message="Updating ubuntu packages"
    echo "${message} ..."

    sudo apt install -y \
        build-essential \
        gcc-multilib \
        dkms \
        apt-transport-https \
        ca-certificates \
        software-properties-common \
        curl \
    && sudo apt-get update
    check_fail_msg "$?" "$message" "" || return 1

    echo "$message: [OK]"
    return 0
}


function check_module(){
    module_name="$1"
    module_exist="$2"

    # If module exists grep will throw an opposite return
    grep -q "^$module_name\ " /proc/modules 2>/dev/null
    if [[ ! "$?" -eq "$module_exist" ]];then
        echo "[OK]"
        return 0
    fi

    echo "[...]"
    return 1
}


function p__driver_check(){
    echo -n "Ensure nouveau driver is disabled: "
    check_module "nouveau" 0 || return 1

    if [ ! -z "$1" ];then
        # driver version check
        DRIVER_VERSION="$1"
        nvidia_driver_check || return 1
    else
        # module check only
        echo -n "Check if NVIDIA driver is loaded: "
        check_module "nvidia" 1 || return 1
    fi

    return 0
}


function p__disable_nouveau(){
    echo -n "Ensure nouveau driver is disabled: "
    check_module "nouveau" 0 && return 0

    echo "Disabling nouveau driver ..."

    echo -e "blacklist nouveau\noptions nouveau modeset=0" |sudo tee /etc/modprobe.d/blacklist-nouveau.conf

    sudo update-initramfs -u; [[ ! "$?" -eq 0 ]] || return 1

    # Reboot automatically in unattended mode
    if [[ "$1" == "unattended" ]];then
        echo "Rebooting ..."
        sudo reboot
        exit 0
    fi

    # Ask if can be rebooted
    while true;do
        read -p "Reboot required. Reboot? " yn
        case $yn in
            [Yy]*) 
                echo "Rebooting ..."
                sudo reboot
                exit 0
                ;;
            [Nn]*) 
                echo "Cancelled reboot."
                return 1
                ;;
            *)  echo "Please answer yes or no."
                ;;
        esac
    done

    return 1
}


function nvidia_driver_check(){
    echo -n "Check if NVIDIA driver is loaded: "
    check_module "nvidia" 1 || return 1

    echo -n "Check if NVIDIA driver is updated: "

    version_installed=$(modinfo nvidia |grep "^version:" |grep -o "[0-9\.]*$")

    if [[ -z "$version_installed" ]];then echo "[...]"; return 1;fi
    if [[ "$DRIVER_VERSION" > "$version_installed" ]];then
        echo "[NO]"
        return 1
    fi
    
    echo "[OK]";
    return 0;
}


function p__driver(){
    [ ! -z "$1" ] && DRIVER_VERSION="$1"
    nvidia_driver_check && return 0

    message="(Re-)install NVIDIA driver"
    echo "${message} ..."

    filename="NVIDIA-Linux-x86_64-$DRIVER_VERSION.run"
    tmpdir="$HOME/tmp"
    fqfn="$tmpdir/$filename"

    echo $fqfn
    if [[ ! -s "$fqfn" ]];then
        [[ ! -d "$tmpdir" ]] && mkdir "$tmpdir"

        echo "Downloading: $filename to $tmpdir/"
        curl --connect-timeout 10 "http://us.download.nvidia.com/XFree86/Linux-x86_64/$DRIVER_VERSION/$filename" -o "$fqfn.tmp"
        return_code="$?"

        if [[ "$return_code" -eq 0 ]];then
            # verify file contents
            head -n1 "$fqfn.tmp" |sed 's/\ //g' |grep -qc '#!/bin/sh'
            return_code="$?"
        fi

        if [[ ! "$return_code" -eq 0 ]];then
            echo "Downloading $filename failed. Exiting."
            [[ -f "$fqfn.tmp" ]] && rm "$fqfn.tmp"
            exit 1
        fi

        mv "$fqfn.tmp" "$fqfn"
    fi

    sudo sh "$fqfn" --silent --dkms --x-library-path=/usr/lib --x-module-path=/usr/lib/xorg/modules
    check_fail_msg "$?" "$message" "" || return 1
    return 0
}


function docker_service_check(){
    message="Check if docker is running: "
    echo -n "$message"
    run_as_docker timeout 10 docker ps -a >/dev/null 2>&1
    check_fail "$?" "" || return 1
    echo "[OK]"
    return 0
}


function p__docker(){
    docker_service_check && return 0

    release=$(lsb_release -cs)

    message="(Re-)installing docker: "
    curl -fsSL "https://download.docker.com/linux/ubuntu/gpg" | sudo apt-key add - \
        && sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu ${release} stable"
    check_fail_msg "$?" "$message" "0" || return 1

    sudo apt-get update && sudo apt-get install -y docker-ce
    check_fail_msg "$?" "$message" "1" || return 1

    #Install latest compose version
    sudo curl -L\
        "https://github.com/docker/compose/releases/download/$DOCKER_COMPOSE_VERSION/docker-compose-`uname -s`-`uname -m`" \
        -o /usr/bin/docker-compose \
        && sudo chmod +x /usr/bin/docker-compose
    check_fail_msg "$?" "$message" "2" || return 1

    sudo usermod -aG docker `whoami` 
    check_fail_msg "$?" "$message" "3" || return 1
    
    echo "${message}: [OK]"
    docker_service_check
    return $?
}


function logout_message(){
    echo "NOTE: group docker not yet active within main shell."
    echo "Logout and (re-)Login to activate group."
    return 0
}


function p__nvidia_docker(){
    message="Installing NVIDIA docker files"
    echo "${message} ..."

    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey |sudo apt-key add -
    check_fail_msg "$?" "$message" "0" || return 1

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
        |sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    check_fail_msg "$?" "$message" "1" || return 1
    
    sudo apt-get update; check_fail_msg "$?" "$message" "2" || return 1
    sudo apt-get install -y nvidia-docker2; check_fail_msg "$?" "$message" "3" || return 1

    echo "${message}: [OK]"
    return 0
}


function p__docker_restart(){
    message="Restarting docker service"

    sudo service docker restart
    check_fail_msg "$?" "$message" "" || return 1
    
    echo "${message}: [OK]"
    return 0
}


function p__check_runtime(){
    message="Testing CUDA runtime"
    echo "$message ..."
    run_as_docker docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
    check_fail_msg "$?" "$message" "" || return 1
    echo "${message}: [OK]"
    return 0
}


function p__help(){
    #fqfn="$1"
    usage
}


function check_distro(){
    echo -n "OS distribution supported: "
    release=$(lsb_release -cs 2>/dev/null || echo "")
    if [[ -z "$release" ]];then
        echo "[FAIL]"
        echo "Redirecting to usage:"
        usage
        return 1
    fi

    case "$release" in
        bionic) ;;
        xenial) ;;
        *)  
            echo "[FAIL]"
            echo "Redirecting to usage:"
            usage
            return 1
            ;;
    esac

    echo "[OK]"
    return 0
}


check_distro || exit 1

INPUT="$1"
case "$INPUT" in
        --server)
            p__ubuntu \
            && p__disable_nouveau "unattended" \
            && p__driver \
            && p__docker \
            && p__nvidia_docker \
            && p__docker_restart

            if [[ ! "$?" -eq 0 ]];then
                echo "Install ($INPUT) complete: [FAIL]"
                exit 1
            fi            

            is_group_docker || logout_message
            echo "Install ($INPUT) complete: [OK]"
            exit 0
            ;;
        --desktop)
            # install excluding the driver parts
            # assuming the driver is installed
            p__driver_check \
            && p__ubuntu \
            && p__docker \
            && p__nvidia_docker \
            && p__docker_restart
            
            if [[ ! "$?" -eq 0 ]];then
                echo "Install ($INPUT) complete: [FAIL]"
                exit 1
            fi            

            is_group_docker || logout_message
            echo "Install ($INPUT) complete: [OK]"
            exit 0
            ;;
        --custom)
            # run one or more functions separately
            shift
            parts=$(filter_input_argstr "$@")
            for part in ${parts[@]};do
                case "$part" in
                    ubuntu)    p__ubuntu;;
                    docker)    p__docker;;
                    nvidia-docker)  p__nvidia_docker;;
                    disable*nouveau)    p__disable_nouveau;;
                    driver) p__driver;;
                    driver=[0-9]*)  p__driver "${part##*=}";;
                    docker-restart) p__docker_restart;;
                    *)  echo "RUNPART unknown: $part"; exit 1;;
                esac

                if [[ ! "$?" -eq 0 ]];then
                    echo "Part: $part [FAILED]"
                    exit 1
                fi
            done
            exit 0
            ;;
        --check)
            # run one or more functions separately
            shift
            parts=$(filter_input_argstr "$@")
            for part in ${parts[@]};do
                case "$part" in
                    driver-check)   p__driver_check;;
                    driver-check=[0-9]*)   p__driver_check "${part##*=}";;
                    runtime)    p__check_runtime;;
                    *)  echo "CHECK unknown: $part"; exit 1;;
                esac

                if [[ ! "$?" -eq 0 ]];then
                    echo "Part: $part [FAILED]"
                    exit 1
                fi
            done
            exit 0
            ;;
        *)
            p__help "$(realpath $0)"
            ;;
esac
