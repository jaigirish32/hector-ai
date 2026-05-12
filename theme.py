"""
Central theme for HECTOR-AI.

Every color, font, and style rule lives here. If the client asks to change
the gold to a different color, or switch to a light theme, you edit this
one file and the whole app updates.
"""
from PySide6.QtWidgets import QApplication


class Colors:
    """Named colors used throughout the app. Accessed as Colors.GOLD etc."""

    # Backgrounds — from deepest to lightest
    BG_SIDEBAR = "#0A0A0A"
    BG_APP = "#0F0F0F"
    BG_CARD = "#161616"
    BG_CARD_HOVER = "#1D1D1D"
    BG_INPUT = "#1A1A1A"
    BG_FOOTER = "#121212"

    # Chips — used for model selector toggles
    BG_CHIP = "#1A1A1A"
    BG_CHIP_ACTIVE = "#0A2A28"  # dark cyan-tinted background when selected

    # Borders
    BORDER = "#242424"
    BORDER_HOVER = "#333333"
    BORDER_ACTIVE = "#D4AF37"

    # Text — three levels of hierarchy
    TEXT_PRIMARY = "#EDEDED"
    TEXT_SECONDARY = "#9A9A9A"
    TEXT_TERTIARY = "#5E5E5E"

    # Brand
    GOLD = "#00D4C4"
    GOLD_HOVER = "#00E6D5"
    GOLD_DARK = "#007D74"
    GOLD_TEXT_ON = "#042926"
    BRAND_GOLD = "#D4AF37"

    # Semantic — status colors
    SUCCESS = "#4ADE80"
    ERROR = "#F87171"
    WARNING = "#FBBF24"

    # Provider accent tints (muted, used inside provider initial badges)
    OPENAI_BG = "#0F2E20"
    OPENAI_FG = "#10A37F"
    GROK_BG = "#1A1A1A"
    GROK_FG = "#EDEDED"
    GEMINI_BG = "#0F1E35"
    GEMINI_FG = "#4A90E2"
    LOCAL_BG = "#1A1A1A"
    LOCAL_FG = "#8A8A8A"


# Font stacks — first available on each OS wins.
FONT_STACK = '"Segoe UI", "SF Pro Display", "Inter", "Helvetica Neue", sans-serif'
FONT_MONO = '"Cascadia Code", "SF Mono", "Consolas", "Menlo", monospace'


def _build_stylesheet() -> str:
    """Assemble the QSS stylesheet from the color constants above."""
    c = Colors  # local alias to keep lines short
    return f"""
    /* ===== Global ===== */
    * {{
        font-family: {FONT_STACK};
        color: {c.TEXT_PRIMARY};
        outline: 0;
    }}

    QMainWindow, QWidget#centralWidget {{
        background-color: {c.BG_APP};
    }}

    /* ===== Sidebar ===== */
    QWidget#sidebar {{
        background-color: {c.BG_SIDEBAR};
        border-right: 1px solid {c.BORDER};
    }}
    QLabel#logo {{
        color: {c.GOLD};
        font-size: 22px;
        font-weight: 600;
        letter-spacing: 1px;
    }}
    QFrame#logoContainer {{
    background-color: #F4F2EC;
    border-radius: 10px;
    border: 1px solid {c.BORDER};
    }}

    QLabel#tagline {{
        color: {c.TEXT_TERTIARY};
        font-size: 11px;
        font-style: italic;
    }}
    QLabel#sectionLabel {{
        color: {c.TEXT_TERTIARY};
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.8px;
        padding-left: 4px;
    }}
    QLabel#brandFooter {{
    color: {c.BRAND_GOLD};
    font-size: 14px;
    font-weight: 500;
    }}

    /* ===== Nav buttons (sidebar items) ===== */
    QPushButton#navButton {{
        background-color: transparent;
        color: {c.TEXT_SECONDARY};
        border: 0;
        border-radius: 8px;
        padding: 10px 12px;
        text-align: left;
        font-size: 13px;
        font-weight: 500;
    }}
    QPushButton#navButton:hover {{
        background-color: {c.BG_CARD};
        color: {c.TEXT_PRIMARY};
    }}
    QPushButton#navButton[active="true"] {{
        background-color: {c.BG_CARD};
        color: {c.GOLD};
        font-weight: 600;
    }}

    /* ===== Cards (generic container) ===== */
    QFrame#card {{
        background-color: {c.BG_CARD};
        border: 1px solid {c.BORDER};
        border-radius: 12px;
    }}
    QFrame#card[highlighted="true"] {{
        border: 1px solid {c.GOLD};
    }}

    /* ===== Card header buttons (copy + stop) =====
       These two share the same slot in the response card header — only
       one is visible at a time. Copy after completion; Stop while
       generating. Both styled identically for visual consistency, with
       the Stop hover taking a destructive-action accent so users feel
       the click weight before pressing.
    */
    QPushButton#copyButton, QPushButton#stopButton {{
        background-color: transparent;
        border: 1px solid {c.BORDER};
        border-radius: 6px;
        padding: 0;
    }}
    QPushButton#copyButton:hover {{
        background-color: {c.BG_CARD_HOVER};
        border: 1px solid {c.BORDER_HOVER};
    }}
    QPushButton#copyButton:disabled {{
        border: 1px solid {c.BORDER};
        background-color: transparent;
    }}
    QPushButton#stopButton:hover {{
        background-color: {c.BG_CARD_HOVER};
        border: 1px solid {c.ERROR};
    }}
    QPushButton#stopButton:disabled {{
        border: 1px solid {c.BORDER};
        background-color: transparent;
    }}

    /* ===== Primary button (gold, for Run / main actions) ===== */
    QPushButton#primary {{
        background-color: {c.GOLD};
        color: {c.GOLD_TEXT_ON};
        border: 0;
        border-radius: 8px;
        padding: 9px 18px;
        font-size: 13px;
        font-weight: 600;
    }}
    /* ===== Model selection chips ===== */
    QPushButton#chip {{
        background-color: {c.BG_CHIP};
        color: {c.TEXT_SECONDARY};
        border: 1px solid {c.BORDER};
        border-radius: 14px;
        padding: 5px 14px;
        font-size: 12px;
        font-weight: 500;
        text-align: center;
    }}
    QPushButton#chip:hover {{
        border: 1px solid {c.BORDER_HOVER};
        color: {c.TEXT_PRIMARY};
    }}
    QPushButton#chip[selected="true"] {{
        background-color: {c.BG_CHIP_ACTIVE};
        color: {c.GOLD};
        border: 1px solid {c.GOLD_DARK};
    }}

    QPushButton#primary:hover {{
        background-color: {c.GOLD_HOVER};
    }}
    QPushButton#primary:pressed {{
        background-color: {c.GOLD_DARK};
    }}
    QPushButton#primary:disabled {{
        background-color: #2A2A2A;
        color: {c.TEXT_TERTIARY};
    }}

    /* ===== Secondary button (outlined) ===== */
    QPushButton#secondary {{
        background-color: transparent;
        color: {c.TEXT_SECONDARY};
        border: 1px solid {c.BORDER};
        border-radius: 8px;
        padding: 7px 14px;
        font-size: 12px;
    }}
    QPushButton#secondary:hover {{
        border: 1px solid {c.BORDER_HOVER};
        color: {c.TEXT_PRIMARY};
    }}

    /* ===== Text inputs ===== */
    QTextEdit, QLineEdit {{
        background-color: {c.BG_INPUT};
        border: 1px solid {c.BORDER};
        border-radius: 8px;
        color: {c.TEXT_PRIMARY};
        padding: 8px 12px;
        font-size: 13px;
        selection-background-color: {c.GOLD_DARK};
    }}
    /* ===== Prompt input (special case — transparent, part of the card) ===== */
    QTextEdit#promptInput {{
        background-color: transparent;
        border: 0;
        color: {c.TEXT_PRIMARY};
        font-size: 14px;
        padding: 0;
        line-height: 1.55;
    }}

    /* ===== Hint text (counters, param strings) ===== */
    QLabel#hintText {{
        color: {c.TEXT_TERTIARY};
        font-size: 11px;
    }}

    /* ===== Attached file chips ===== */
    QFrame#fileChip {{
        background-color: {c.BG_INPUT};
        border: 1px solid {c.BORDER};
        border-radius: 6px;
    }}
    QLabel#fileChipName {{
        color: {c.TEXT_SECONDARY};
        font-size: 11px;
    }}
    QLabel#fileChipSize {{
        color: {c.TEXT_TERTIARY};
        font-size: 10px;
    }}
    QPushButton#fileChipRemove {{
        background-color: transparent;
        color: {c.TEXT_TERTIARY};
        border: 0;
        border-radius: 9px;
        font-size: 14px;
        font-weight: 400;
        padding: 0;
    }}
    QPushButton#fileChipRemove:hover {{
        background-color: {c.BG_CARD_HOVER};
        color: {c.ERROR};
    }}

    QTextEdit:focus, QLineEdit:focus {{
        border: 1px solid {c.GOLD_DARK};
    }}

    /* ===== Dialog / popup styling (QDialog, QMessageBox) =====
       Same problem as the QMenu issue documented above: without
       explicit rules, popups inherit the global `*` rule's color
       (light) but get the OS-default background (white on macOS).
       Result: dark text on white = unreadable error dialogs.

       These rules apply to every QDialog and QMessageBox in the
       app — confirmation dialogs (delete a file), error popups
       (upload failed), warnings, info boxes. They cascade to
       QLabel and QPushButton inside the dialog so we don't have
       to style each child separately. v0.1.9 added this section.
    */
    QDialog, QMessageBox {{
        background-color: {c.BG_CARD};
        color: {c.TEXT_PRIMARY};
    }}
    QDialog QLabel, QMessageBox QLabel {{
        background-color: transparent;
        color: {c.TEXT_PRIMARY};
        font-size: 13px;
    }}
    QMessageBox QPushButton {{
        background-color: {c.BG_INPUT};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        border-radius: 6px;
        padding: 6px 18px;
        font-size: 12px;
        font-weight: 500;
        min-width: 70px;
    }}
    QMessageBox QPushButton:hover {{
        background-color: {c.BG_CARD_HOVER};
        border: 1px solid {c.BORDER_HOVER};
    }}
    QMessageBox QPushButton:default {{
        background-color: {c.GOLD};
        color: {c.GOLD_TEXT_ON};
        border: 1px solid {c.GOLD_DARK};
    }}
    QMessageBox QPushButton:default:hover {{
        background-color: {c.GOLD_HOVER};
    }}

    /* ===== Tooltips =====
       Tooltips appear on hover (Run button, copy button, file chips).
       macOS defaults to a yellow-ish background that clashes with
       the dark theme. Force dark surfaces here too.
    */
    QToolTip {{
        background-color: {c.BG_CARD};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 11px;
    }}

    /* ===== Defensive: standard form controls =====
       These widgets might appear in current or future settings
       dialogs, and they all have the same fall-through-to-system
       behavior on macOS. Pre-empting the bug rather than waiting
       to be bitten by it.
    */
    QComboBox {{
        background-color: {c.BG_INPUT};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        border-radius: 6px;
        padding: 5px 10px;
        font-size: 13px;
    }}
    QComboBox:hover {{
        border: 1px solid {c.BORDER_HOVER};
    }}
    QComboBox QAbstractItemView {{
        background-color: {c.BG_CARD};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        selection-background-color: {c.BG_CARD_HOVER};
    }}

    QCheckBox, QRadioButton {{
        color: {c.TEXT_PRIMARY};
        font-size: 13px;
        spacing: 8px;
    }}
    QCheckBox::indicator, QRadioButton::indicator {{
        background-color: {c.BG_INPUT};
        border: 1px solid {c.BORDER};
        width: 16px;
        height: 16px;
    }}
    QCheckBox::indicator {{
        border-radius: 3px;
    }}
    QRadioButton::indicator {{
        border-radius: 8px;
    }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background-color: {c.GOLD};
        border: 1px solid {c.GOLD_DARK};
    }}

    QSpinBox, QDoubleSpinBox {{
        background-color: {c.BG_INPUT};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        border-radius: 6px;
        padding: 5px 8px;
        font-size: 13px;
    }}

    QGroupBox {{
        background-color: transparent;
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        border-radius: 8px;
        margin-top: 12px;
        padding-top: 8px;
    }}
    QGroupBox::title {{
        color: {c.TEXT_SECONDARY};
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
        font-size: 11px;
        font-weight: 600;
    }}

    QHeaderView::section {{
        background-color: {c.BG_CARD};
        color: {c.TEXT_SECONDARY};
        border: 0;
        border-bottom: 1px solid {c.BORDER};
        padding: 6px 10px;
        font-size: 11px;
        font-weight: 600;
    }}

    QTableView, QListView, QTreeView {{
        background-color: {c.BG_APP};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        border-radius: 6px;
        selection-background-color: {c.BG_CARD_HOVER};
        selection-color: {c.TEXT_PRIMARY};
        outline: 0;
    }}
    QTableView::item, QListView::item, QTreeView::item {{
        padding: 4px;
    }}

    QProgressBar {{
        background-color: {c.BG_INPUT};
        border: 1px solid {c.BORDER};
        border-radius: 6px;
        text-align: center;
        color: {c.TEXT_PRIMARY};
        font-size: 11px;
    }}
    QProgressBar::chunk {{
        background-color: {c.GOLD};
        border-radius: 5px;
    }}

    /* ===== Scroll areas — defensive Mac fix =====
       On macOS, QScrollArea and QAbstractScrollArea-derived widgets
       inherit a white background by default unless given an explicit
       rule. Windows defaults dark-ish so the bug is invisible there.
       These rules force transparent on every scroll area; targeted
       rules below explicitly paint the sidebar background where
       transparency isn't enough (the FILES list).
    */
    QScrollArea {{
        background-color: transparent;
        border: 0;
    }}

    /* Compare view's card scroll area — paint with app background
       so on Mac the area behind cards doesn't show white when a card
       is hidden by chip toggle. Targeted by object name so we don't
       affect FILES panel which uses its own BG_SIDEBAR rule. */
    QScrollArea#cardsScroll,
    QScrollArea#cardsScroll > QWidget,
    QWidget#cardsContainer {{
        background-color: {c.BG_APP};
        border: 0;
    }}

    QAbstractScrollArea {{
        background-color: transparent;
    }}

    /* ===== File library panel (sidebar FILES section) ===== */
    QScrollArea#filesRowsScroll {{
        background-color: {c.BG_SIDEBAR};
        border: 0;
    }}
    QWidget#filesRowsContainer {{
        background-color: {c.BG_SIDEBAR};
    }}

    /* ===== Scrollbars (subtle, minimal) ===== */
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {c.BORDER};
        border-radius: 4px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {c.BORDER_HOVER};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: transparent;
    }}

    /* ===== Context menus (right-click Cut/Copy/Paste etc.) =====
       Without explicit QMenu rules, the global `*` rule's color leaked
       into Qt's default menu, and on macOS the system menu background
       defaults to white — light text on white = invisible items until
       hovered. These rules force a dark background everywhere.
       v0.1.4 added this section.
    */
    QMenu {{
        background-color: {c.BG_CARD};
        color: {c.TEXT_PRIMARY};
        border: 1px solid {c.BORDER};
        border-radius: 6px;
        padding: 4px 0;
    }}
    QMenu::item {{
        background-color: transparent;
        color: {c.TEXT_PRIMARY};
        padding: 6px 24px 6px 16px;
        font-size: 13px;
    }}
    QMenu::item:selected {{
        background-color: {c.BG_CARD_HOVER};
        color: {c.TEXT_PRIMARY};
    }}
    QMenu::item:disabled {{
        color: {c.TEXT_TERTIARY};
    }}
    QMenu::separator {{
        height: 1px;
        background-color: {c.BORDER};
        margin: 4px 8px;
    }}

    /* ===== Settings view — header ===== */
    QFrame#settingsHeader {{
        background-color: {c.BG_SIDEBAR};
        border: 0;
        border-bottom: 1px solid {c.BORDER};
    }}
    QLabel#settingsTitle {{
        color: {c.TEXT_PRIMARY};
        font-size: 22px;
        font-weight: 600;
    }}
    QLabel#settingsSubtitle {{
        color: {c.TEXT_SECONDARY};
        font-size: 12px;
    }}
    QLabel#readinessLabel {{
        color: {c.TEXT_TERTIARY};
        font-size: 12px;
        font-weight: 500;
        padding-top: 6px;
    }}
    QLabel#readinessLabelGood {{
        color: {c.SUCCESS};
        font-size: 12px;
        font-weight: 500;
        padding-top: 6px;
    }}
    /* ===== Settings view rows ===== */
    QFrame#settingsRow {{
        background-color: {c.BG_CARD};
        border: 1px solid {c.BORDER};
        border-radius: 10px;
    }}
    QLabel#fieldLabel {{
        color: {c.TEXT_PRIMARY};
        font-size: 13px;
        font-weight: 600;
    }}
    QLabel#fieldHelper {{
        color: {c.TEXT_TERTIARY};
        font-size: 11px;
    }}
    QLabel#fieldStatus {{
        color: {c.TEXT_TERTIARY};
        font-size: 11px;
        font-weight: 500;
    }}
    QLabel#fieldStatusGood {{
        color: {c.SUCCESS};
        font-size: 11px;
        font-weight: 500;
    }}
    QLabel#fieldStatusError {{
        color: {c.ERROR};
        font-size: 11px;
        font-weight: 500;
    }}
    QLineEdit#secretInput {{
        background-color: {c.BG_INPUT};
        border: 1px solid {c.BORDER};
        border-radius: 7px;
        padding: 7px 12px;
        font-size: 13px;
        font-family: {FONT_MONO};
        color: {c.TEXT_PRIMARY};
        selection-background-color: {c.GOLD_DARK};
    }}
    QLineEdit#secretInput:focus {{
        border: 1px solid {c.GOLD_DARK};
    }}
    """


def apply_theme(app: QApplication) -> None:
    """Apply the global stylesheet to a QApplication instance."""
    app.setStyleSheet(_build_stylesheet())