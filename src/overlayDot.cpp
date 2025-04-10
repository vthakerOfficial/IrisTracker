#include <overlayDot.h>

DotOverlay::DotOverlay() 
: 
m_handleDotOverlay(nullptr),
m_threadMsgID(0)
{}
DotOverlay::~DotOverlay() {
    m_shouldClose.store(true);

    if(m_threadMsgID != 0) {
        PostThreadMessage(m_threadMsgID, WM_QUIT, 0, 0);
    }

    if (m_threadMsg.joinable()) {
        m_threadMsg.join();
    }
    std::cout << "Destroyed dot overlay\n";
}

void DotOverlay::runDotOverlay(int nCmdShow)
{
    HINSTANCE hInstance = GetModuleHandle(NULL);

    m_threadMsg = std::thread([this, hInstance, nCmdShow]() {
        
        m_threadMsgID = GetCurrentThreadId();

        WNDCLASSEX wc = { 0 };
        wc.lpszClassName = m_CLASS_NAME;
        wc.cbSize = sizeof(WNDCLASSEX);
        wc.hbrBackground = NULL;
        wc.lpfnWndProc = WndProc;
        wc.style = CS_HREDRAW | CS_VREDRAW;
        wc.hInstance = hInstance;
        wc.hCursor = LoadCursor(NULL, IDC_ARROW);

        if (!RegisterClassEx(&wc)) {
            std::cout << "Couldnt register window (dot overlay) class\n";
            return;
        }
        int x = 200, y = 200;

        m_handleDotOverlay = CreateWindowEx(
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_NOACTIVATE, // allows keyboard mouse etc passthru
            m_CLASS_NAME,
            TEXT("DotOverlay"),
            WS_POPUP,
            x,
            y,
            m_radius,
            m_radius,
            NULL, 
            NULL, 
            hInstance, 
            NULL
        );

        if (!m_handleDotOverlay) {
            std::cout << "Failedto create overlay windows (for dot ovleray)";
            return;
        }

        // make win transparent + allow clicks to go thru
        SetLayeredWindowAttributes(m_handleDotOverlay, 0, 255, LWA_ALPHA);

        ShowWindow(m_handleDotOverlay, nCmdShow);
        UpdateWindow(m_handleDotOverlay);

        // allowing main thread to continue by setting bool to true
        m_bInitialized.store(true, std::memory_order_release);

        MSG msg;
        while (!m_shouldClose.load() && GetMessage(&msg, NULL, 0, 0)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        if (m_handleDotOverlay) {
            DestroyWindow(m_handleDotOverlay);
            m_handleDotOverlay = nullptr;
        }
        UnregisterClass(m_CLASS_NAME, hInstance);
    });

    // ensures dot window is initialized before continuing main thread
    while (!m_bInitialized.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}



LRESULT CALLBACK DotOverlay::WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hWnd, &ps);
        HBRUSH brush = CreateSolidBrush(RGB(255, 0, 0));
        HBRUSH oldBrush  = (HBRUSH)SelectObject(hdc, brush);
        Ellipse(hdc, 0, 0, m_radius, m_radius);
        SelectObject(hdc, oldBrush);
        DeleteObject(brush);
        EndPaint(hWnd, &ps);
    }
    break;

    case WM_DESTROY:
        PostQuitMessage(0);
        break;

    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

void DotOverlay::moveTo(int x, int y)
{
    SetWindowPos(m_handleDotOverlay,HWND_TOPMOST,x,y,m_radius,m_radius,SWP_NOACTIVATE | SWP_SHOWWINDOW);
}
