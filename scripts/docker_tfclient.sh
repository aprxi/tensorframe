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


PROGNAME="docker_tfclient.sh"

PROJECT_ROOT="$(dirname $(dirname $(realpath -s $0)))"
HOST_UID=$(id -u)
HOST_GID=$(id -g)
HOST_USER=$(id -un)
HOST_GROUP=$(id -gn)
DOCKER_RUNTIME=

function usage(){
    cat << USAGE
  Notes
    $PROGNAME prepares a development version of Tensorframe. We input
    the default Dockerfile (used for tfclient) and complement this with
    development parts. This allows for rapid testing and prototyping of
    new features and/ or updated functionality.

    Note. A GPU enabled system is not required to build, but it is
    needed for all tests to complete.
    
  Usage:
    # (re-)build running container service for tfclient
    command:    $PROGNAME [COMMAND] [MODE]
    examples:
        $PROGNAME --build user
        $PROGNAME --generate user

  COMMANDS
    --build     (Re-)generate docker files and (re-)build container service
    --generate  Only (re-)generate docker files in build/

  MODES
    user        
    developer

USAGE
    return 0
}


function p__help(){
    usage
    return 0
}


function _err(){
    echo "${PROGNAME}::$1"
    exit 1
}


function _err_help(){
    usage
    _err "$1"   
}


function create_compose_minimal(){
    [[ -z "$1" ]] && _err "create_compose_minimal(): empty argument (1)"

    composefile="$1"

    cat > "$composefile" << COMPOSE
{
    "version": "2.3",
    "services": {
        "tfclient": {
            "container_name": "tfclient",
            "hostname": "tfclient",
            "command": ["sh", "/bin/entrypoint.sh"]
        }
    }
}
COMPOSE
    return $?
}


function docker_profile__developer(){
    [[ -z "$1" ]] && _err "docker_profile__developer(): empty argument (2)"

    outputfile="$1"

    cat >>"$outputfile" << PROFILE
FROM scratch as developer-latest
ENV CUDA_VERSION 10.0.130
ENV CUDA_PKG_VERSION 10-0=\$CUDA_VERSION-1
COPY --from=user-latest . .
RUN apt-get update --fix-missing \\
    && apt-get -y install \\
        build-essential \\
        cuda-minimal-build-\$CUDA_PKG_VERSION \\
        cuda-cublas-dev-\$CUDA_PKG_VERSION \\
        cuda-curand-dev-\$CUDA_PKG_VERSION \\
        python3-pip \\
        git \\
        sudo \\
        vim 
RUN chown -R ${HOST_USER}:${HOST_USER} /var/run/supervisor/. /service/${HOST_USER}/. \\
    && (umask 0227 && echo "%${HOST_USER}  ALL=(ALL) NOPASSWD: ALL" >/etc/sudoers.d/${HOST_USER}) \\
    && python3 -m pip install nose && ln -sf nosetests /usr/local/bin/nosetests3
PROFILE
    return $?
}


function docker_profile__user(){
    [[ -z "$1" ]] && _err "docker_profile__user(): empty argument (1)"
    [[ -z "$2" ]] && _err "docker_profile__user(): empty argument (2)"

    inputfile="$1"
    outputfile="$2"

    [[ ! -s "$inputfile" ]] && _err "docker_profile__user(): inputfile empty or does not exist: ${inputfile}"
    
    cat "$inputfile" >"$outputfile"
    cat >>"$outputfile" << PROFILE
FROM scratch as user-latest
COPY --from=base . .
RUN groupadd -r ${HOST_GROUP} -g ${HOST_GID} \\
    && useradd -d /service/${HOST_USER} -m -r -l -N -s /usr/sbin/nologin \\
        -u ${HOST_UID} -g ${HOST_GID} ${HOST_USER} \\
    && sed -i "s:{{ *HOST_USER *}}:${HOST_USER}:g" /etc/supervisor/conf.d/jupyter.conf \\
    && chown -R ${HOST_USER}:${HOST_USER} /var/run/supervisor/.
PROFILE
    return $?
}


function create_dockerfile_local(){
    [[ -z "$1" ]] && _err "create_dockerfile_local(): empty argument (1)"
    [[ -z "$2" ]] && _err "create_dockerfile_local(): empty argument (2)"
    
    profile_tag="$1"
    builddir="$2"

    cd "$PROJECT_ROOT" || _err "Cant cd into: ${PROJECT_ROOT}"

    [[ ! -d "$builddir" ]] && _err "create_dockerfile_local(): builddir does not exist: ${builddir}"

    dockerfile="${PROJECT_ROOT}/Dockerfile"
    dockerfile_target="${builddir}/Dockerfile.${profile_tag}-${HOST_USER}"

    case "$profile_tag" in
        user)
            docker_profile__user "$dockerfile" "$dockerfile_target"
            ;;
        developer)
            docker_profile__user "$dockerfile" "${dockerfile_target}.tmp" "$profile_tag"
            docker_profile__developer "${dockerfile_target}.tmp" && mv "${dockerfile_target}.tmp" "$dockerfile_target"
            ;;
        *)  _err "create_compose_local(): invalid profile: ${profile_tag}"
            ;;
    esac
}


function create_compose_local(){
    [[ -z "$1" ]] && _err "create_compose_local(): empty argument (1)"
    [[ -z "$2" ]] && _err "create_compose_local(): empty argument (2)"

    profile_tag="$1"
    builddir="$2"
    
    cd "$PROJECT_ROOT" || _err "Cant cd into: ${PROJECT_ROOT}"
    
    [[ ! -d "$builddir" ]] && _err "create_compose_local(): builddir does not exist: ${builddir}"
    
    [[ -s "$PROJECT_ROOT/.git/config" ]] || _err "Expected a GIT directory: ${PROJECT_ROOT}"
    
    composefile="${builddir}/docker-compose.${profile_tag}-${HOST_USER}.json"
    create_compose_minimal "$composefile" || _err "p__generate_docker(): cant create composefile: ${composefile}"
    
    datadir="$HOME/.tensorframe"
    if [[ ! -d "$datadir" ]];then
        mkdir -p "$datadir" || _err "p__generate_docker(): failed to create datadir: ${datadir}"
    fi

    python3 -c 'import sys, json; json.dump(json.load(sys.stdin), sys.stdout, indent=4)' < "$composefile" \
        |jq '
            .services.tfclient.user = "'"$HOST_UID:$HOST_GID"'" |
            .services.tfclient.image = "'"${HOST_USER}"'/tensorframe_tfclient:'"$profile_tag"'-latest" |
            .services.tfclient.build.dockerfile = "build/Dockerfile.'${profile_tag}-${HOST_USER}'" |
            .services.tfclient.build.context = "../" |
            .services.tfclient.build.target = "'$profile_tag'-latest" |
            .services.tfclient.volumes = [
                {"type": "bind", "source": "'"$datadir"'", "target": "/service/'"${HOST_USER}"'"}] |
            .services.tfclient.ports = ["8888:8888"]
        ' >"${composefile}.tmp" && mv "${composefile}.tmp" "$composefile"

    if [[ ! -z "$DOCKER_RUNTIME" ]];then
        python3 -c 'import sys, json; json.dump(json.load(sys.stdin), sys.stdout, indent=4)' < "${composefile}" \
            |jq '
                .services.tfclient.runtime = "'"$DOCKER_RUNTIME"'" |
                .services.tfclient.environment[.services.tfclient.environment| length] |= .
                    + "DOCKER_RUNTIME='"${DOCKER_RUNTIME}"'"
            ' >"${composefile}.tmp" && mv "${composefile}.tmp" "$composefile"
    else
        python3 -c 'import sys, json; json.dump(json.load(sys.stdin), sys.stdout, indent=4)' < "${composefile}" \
            |jq '
                .services.tfclient.environment[.services.tfclient.environment| length] |= . + "DOCKER_RUNTIME="
            ' >"${composefile}.tmp" && mv "${composefile}.tmp" "$composefile"
    fi
 
    case "$profile_tag" in
        developer)
            python3 -c 'import sys, json; json.dump(json.load(sys.stdin), sys.stdout, indent=4)' < "$composefile" \
                |jq '
                    .services.tfclient.environment[.services.tfclient.environment| length] |= .
                        + "DEVELOPER_MODE=1" |
                    .services.tfclient.volumes[.services.tfclient.volumes| length] |= .
                        + {"type": "bind", "source": "'"$PROJECT_ROOT"'", "target": "/git/tensorframe"}
                ' >"${composefile}.tmp" && mv "${composefile}.tmp" "$composefile"
            ;; 
        *)  ;;
    esac 

    return 0
}


function p__build_container(){
    [[ -z "$1" ]] && _err "p__build_container(): empty argument (1)"

    profile_tag="$1"

    cd "$PROJECT_ROOT" || _err "Cant cd into: ${PROJECT_ROOT}"
    [[ ! -d "./build" ]] && _err "No builddir found in project_root: ${PROJECT_ROOT}"

    composefile="./build/docker-compose.${profile_tag}-${HOST_USER}.json"
    [[ ! -s "$composefile" ]] && _err "p__build_container(): missing composefile: ${composefile}-${HOST_USER}"

    dockerfile="./build/Dockerfile.${profile_tag}-${HOST_USER}"
    [[ ! -s "$dockerfile" ]] && _err "p__build_container(): missing Dockerfile: ${dockerfile}-${HOST_USER}"

    docker-compose -f "$composefile" up -d --build
    [[ ! "$?" -eq 0 ]] && _err "p__build_container(): docker-compose failed for: ${composefile}"

    echo "--------------------------------------------------------------------"
    echo "# Container \"tfclient\" succesfully build for $profile_tag: $HOST_USER"

    if [[ -z "$DOCKER_RUNTIME" ]];then
        echo
        echo "WARNING: GPU NOT DETECTED. CUDA RUNTIME DISABLED."
        echo "Container runs in limited functionality mode."
        echo
    fi
    
    echo "# Directories mapped into container \"tfclient\":"
    echo "#   ${HOME}/.tensorframe -> /service/tfclient"

    case "$profile_tag" in
        developer)  echo "#   ${PROJECT_ROOT} -> /git/tensorframe";;
        *)  ;;
    esac

    echo "# Enter container: docker exec -it tfclient /bin/bash"
    echo "--------------------------------------------------------------------"

    return 0
}


function p__generate_docker(){
    [[ -z "$1" ]] && _err "p__generate_docker(): empty argument (1)"

    profile_tag="$1"
    
    builddir="$PROJECT_ROOT/build"
    if [[ ! -d "$builddir" ]];then
        mkdir -p "$builddir" || _err "p__generate_docker(): failed to create builddir: ${builddir}"
    fi

    create_dockerfile_local "$profile_tag" "$builddir"
    create_compose_local "$profile_tag" "$builddir"
    return 0
}

[[ -z "$1" ]] && _err_help "argument required: [COMMAND] [MODE]"
[[ -z "$2" ]] && _err_help "argument required: [MODE]"

COMMAND="$1"
MODE="$2"
DOCKER_RUNTIME=""

# verify mode input before continuing
case "$MODE" in
    user|developer)
        (exec bash "${PROJECT_ROOT}/scripts/prepare_nvdock.sh" --check driver,runtime)
        [[ "$?" -eq 0 ]] && DOCKER_RUNTIME="nvidia"
        ;;
    *)  _err_help "invalid argument for [MODE]: ${MODE}"
esac


case "$COMMAND" in
    --build)
        p__generate_docker "$MODE" \
        && p__build_container "$MODE" \
        && exit 0

        _err "--build failed"
        ;;
    --generate)
        p__generate_docker "$MODE" \
        && exit 0
        
        _err "--generate failed"
        ;;
    --help)
        p__help && exit 0
        ;;
    *)  
        _err_help "invalid argument for [COMMAND]: ${COMMAND}"
        ;;
esac

# unexpected exit
exit 1
