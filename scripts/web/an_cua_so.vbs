' LiverCancer AI: run one PowerShell script of this folder fully hidden (no console window flash), used by Task Scheduler.
' Usage: wscript.exe an_cua_so.vbs <script.ps1> [arguments...]
Dim sh, fso, dir, cmd, i
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & dir & "\" & WScript.Arguments(0) & """"
For i = 1 To WScript.Arguments.Count - 1
  cmd = cmd & " " & WScript.Arguments(i)
Next
sh.Run cmd, 0, False
