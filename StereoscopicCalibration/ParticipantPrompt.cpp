#include "ParticipantPrompt.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

namespace {

    constexpr int ID_EDIT = 101;
    constexpr int ID_PREVIEW = 102;
    constexpr wchar_t CLASS_NAME[] = L"ParticipantIdPromptWindow";
    constexpr int MAX_ID_INPUT_CHARS = 64;

    struct PromptState {
        HWND edit = nullptr;
        HWND preview = nullptr;
        HWND cancelButton = nullptr;
        HFONT font = nullptr;
        int dpi = 96;

        std::string result;
        bool accepted = false;
        bool done = false;
    };

    // the text currently in the edit box, sanitized
    std::string sanitizedEditText(const PromptState& st)
    {
        wchar_t buf[MAX_ID_INPUT_CHARS + 1] = {};
        GetWindowTextW(st.edit, buf, MAX_ID_INPUT_CHARS + 1);

        std::string narrow; // non-ASCII characters would be removed by the sanitizer anyway
        for (const wchar_t* p = buf; *p; ++p)
            if (*p < 128) narrow.push_back(static_cast<char>(*p));

        return sanitizeParticipantId(narrow);
    }

    // live preview, so it is obvious what the file name will contain
    void updatePreview(const PromptState& st)
    {
        const std::string id = sanitizedEditText(st);
        const std::wstring text = L"Will be saved as:  " + std::wstring(id.begin(), id.end());
        SetWindowTextW(st.preview, text.c_str());
    }

    void tryAccept(HWND hwnd, PromptState& st)
    {
        const std::string id = sanitizedEditText(st);
        if (id.empty()) {
            MessageBoxW(hwnd, L"Please enter a participant ID containing at least one letter or number.",
                L"Participant ID", MB_OK | MB_ICONWARNING);
            SetFocus(st.edit);
            return;
        }

        st.result = id;
        st.accepted = true;
        DestroyWindow(hwnd);
    }

    HWND addChild(HWND parent, const PromptState& st, DWORD exStyle, const wchar_t* cls, const wchar_t* text,
        DWORD style, int x, int y, int w, int h, int id)
    {
        auto px = [&](int v) { return MulDiv(v, st.dpi, 96); };
        HWND c = CreateWindowExW(exStyle, cls, text, WS_CHILD | WS_VISIBLE | style,
            px(x), px(y), px(w), px(h), parent, reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
            GetModuleHandleW(nullptr), nullptr);
        SendMessageW(c, WM_SETFONT, reinterpret_cast<WPARAM>(st.font), TRUE);
        return c;
    }

    LRESULT CALLBACK promptProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp)
    {
        if (msg == WM_NCCREATE) {
            auto* cs = reinterpret_cast<CREATESTRUCTW*>(lp);
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(cs->lpCreateParams));
            return DefWindowProcW(hwnd, msg, wp, lp);
        }

        PromptState* st = reinterpret_cast<PromptState*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        if (!st) return DefWindowProcW(hwnd, msg, wp, lp);

        switch (msg) {
        case WM_CREATE:
            addChild(hwnd, *st, 0, L"STATIC", L"Enter the participant ID:", SS_LEFT, 16, 14, 348, 20, -1);
            st->edit = addChild(hwnd, *st, WS_EX_CLIENTEDGE, L"EDIT", L"", WS_TABSTOP | ES_AUTOHSCROLL,
                16, 38, 348, 26, ID_EDIT);
            st->preview = addChild(hwnd, *st, 0, L"STATIC", L"", SS_LEFT, 16, 72, 348, 20, ID_PREVIEW);
            addChild(hwnd, *st, 0, L"BUTTON", L"OK", WS_TABSTOP | BS_DEFPUSHBUTTON, 190, 106, 84, 28, IDOK);
            st->cancelButton = addChild(hwnd, *st, 0, L"BUTTON", L"Cancel", WS_TABSTOP | BS_PUSHBUTTON,
                280, 106, 84, 28, IDCANCEL);

            SendMessageW(st->edit, EM_LIMITTEXT, MAX_ID_INPUT_CHARS, 0);
            updatePreview(*st);
            return 0;

        case WM_COMMAND:
            if (LOWORD(wp) == ID_EDIT && HIWORD(wp) == EN_CHANGE) { updatePreview(*st); return 0; }
            if (LOWORD(wp) == IDOK) { tryAccept(hwnd, *st); return 0; }
            if (LOWORD(wp) == IDCANCEL) { DestroyWindow(hwnd); return 0; }
            break;

        case WM_CLOSE:
            DestroyWindow(hwnd);
            return 0;

        case WM_DESTROY:
            st->done = true; // ends the loop in ask(); no PostQuitMessage, so no stray WM_QUIT is left behind
            return 0;
        }
        return DefWindowProcW(hwnd, msg, wp, lp);
    }

} // namespace

bool ParticipantPrompt::ask(std::string& participantIdOut)
{
    PromptState st;
    st.dpi = static_cast<int>(GetDpiForSystem()); // 96 unless the process is DPI aware (GLFW makes it so)

    HINSTANCE inst = GetModuleHandleW(nullptr);

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = promptProc;
    wc.hInstance = inst;
    wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512)); // IDC_ARROW (the macro itself breaks non-Unicode builds)
    wc.hbrBackground = reinterpret_cast<HBRUSH>(COLOR_BTNFACE + 1);
    wc.lpszClassName = CLASS_NAME;
    RegisterClassExW(&wc);

    st.font = CreateFontW(-MulDiv(9, st.dpi, 72), 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");

    // 380 x 150 client area (at 96 dpi), centred on the primary monitor, above everything else
    const DWORD style = WS_CAPTION | WS_SYSMENU;
    const DWORD exStyle = WS_EX_TOPMOST;
    RECT rc{ 0, 0, MulDiv(380, st.dpi, 96), MulDiv(150, st.dpi, 96) };
    AdjustWindowRectExForDpi(&rc, style, FALSE, exStyle, static_cast<UINT>(st.dpi));
    const int w = rc.right - rc.left;
    const int h = rc.bottom - rc.top;
    const int x = (GetSystemMetrics(SM_CXSCREEN) - w) / 2;
    const int y = (GetSystemMetrics(SM_CYSCREEN) - h) / 2;

    HWND hwnd = CreateWindowExW(exStyle, CLASS_NAME, L"Participant ID", style, x, y, w, h,
        nullptr, nullptr, inst, &st);

    if (hwnd) {
        ShowWindow(hwnd, SW_SHOW);
        SetForegroundWindow(hwnd);
        SetFocus(st.edit);

        MSG msg;
        while (!st.done && GetMessageW(&msg, nullptr, 0, 0) > 0) {
            // Enter = OK, Esc = Cancel (Enter on the Cancel button cancels)
            if (msg.message == WM_KEYDOWN && (msg.hwnd == hwnd || IsChild(hwnd, msg.hwnd))) {
                if (msg.wParam == VK_RETURN) {
                    const int id = (GetFocus() == st.cancelButton) ? IDCANCEL : IDOK;
                    SendMessageW(hwnd, WM_COMMAND, id, 0);
                    continue;
                }
                if (msg.wParam == VK_ESCAPE) {
                    SendMessageW(hwnd, WM_COMMAND, IDCANCEL, 0);
                    continue;
                }
            }
            if (IsDialogMessageW(hwnd, &msg)) continue; // Tab between the controls
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }

    if (st.font) DeleteObject(st.font);
    UnregisterClassW(CLASS_NAME, inst);

    if (st.accepted) participantIdOut = st.result;
    return st.accepted;
}