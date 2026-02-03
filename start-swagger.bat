@echo off
chcp 65001 > nul
title CLARA-SSoT Swagger UI Launcher

echo ================================================================
echo           CLARA-SSoT Swagger UI Launcher
echo ================================================================
echo.

REM 현재 스크립트가 있는 디렉토리로 이동
cd /d "%~dp0"

REM docker compose 명령어 확인 (v2 우선, v1 폴백)
docker compose version >nul 2>&1
if %errorlevel% equ 0 (
    set DOCKER_COMPOSE=docker compose
) else (
    docker-compose version >nul 2>&1
    if %errorlevel% equ 0 (
        set DOCKER_COMPOSE=docker-compose
    ) else (
        echo [ERROR] Docker Compose를 찾을 수 없습니다.
        echo Docker Desktop이 설치되어 있고 실행 중인지 확인하세요.
        pause
        exit /b 1
    )
)

echo [1/3] Docker Compose로 서비스 시작 중...
echo      사용 명령어: %DOCKER_COMPOSE%
echo.
%DOCKER_COMPOSE% up -d --build
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Docker Compose 실행에 실패했습니다.
    echo 위 로그에서 빌드 오류 내용을 확인하세요. (Docker Desktop 실행 여부 포함)
    pause
    exit /b 1
)

echo.
echo [2/3] 서비스가 준비될 때까지 대기 중...

REM 서비스가 실제로 응답할 때까지 대기 (최대 60초)
set /a count=0
:wait_loop
if %count% geq 30 (
    echo.
    echo [WARNING] 60초 대기 후에도 서비스가 응답하지 않습니다.
    echo 브라우저에서 직접 확인해 주세요.
    goto open_browser
)
curl -s -o nul -w "%%{http_code}" http://localhost:8000/docs | findstr "200" >nul 2>&1
if %errorlevel% equ 0 goto open_browser
set /a count+=1
timeout /t 2 /nobreak > nul
goto wait_loop

:open_browser
echo.
echo [3/3] Swagger UI 열기...
start http://localhost:8000/docs

echo.
echo Swagger UI가 브라우저에서 열렸습니다!
echo.
echo 서비스 종료하려면 아무 키나 누르세요..
pause > nul

echo.
echo 서비스 종료 중...
%DOCKER_COMPOSE% down

echo.
echo 모든 서비스가 종료되었습니다.
timeout /t 3 > nul
