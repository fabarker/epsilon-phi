
    tell application "Microsoft Excel"
        tell active sheet of active workbook
            set myRange to range "B5:H50"
            set horizontal alignment of myRange to center across selection
        end tell
    end tell
    