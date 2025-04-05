#pragma once

#include <windows.h>
#include <dwmapi.h>
#include <iostream>
#include <thread>
#include <atomic>

#pragma comment(lib, "dwmapi.lib")

class DotOverlay {
public:   
    DotOverlay();
    ~DotOverlay();

    int getScreenWidth()  { return GetSystemMetrics(SM_CXSCREEN); }
    int getScreenHeight() { return GetSystemMetrics(SM_CYSCREEN); }
    
    void moveTo(int x, int y);
        
    void runDotOverlay(int nCmdShow = SW_SHOW);
private:
    static LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam);

private:
    const TCHAR* m_CLASS_NAME = TEXT("DotOverlayClass");
    static constexpr const int m_radius = 50;
    HWND m_handleDotOverlay = nullptr;

    std::thread m_threadMsg;
    DWORD m_threadMsgID = 0;
    std::atomic<bool> m_shouldClose { false };
};