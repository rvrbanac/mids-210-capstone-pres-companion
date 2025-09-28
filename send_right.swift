// send_right.swift
import Foundation
import ApplicationServices

func sendKey(_ key: CGKeyCode) {
    guard let src = CGEventSource(stateID: .hidSystemState) else { return }
    let down = CGEvent(keyboardEventSource: src, virtualKey: key, keyDown: true)!
    let up   = CGEvent(keyboardEventSource: src, virtualKey: key, keyDown: false)!
    down.post(tap: .cghidEventTap)
    up.post(tap: .cghidEventTap)
}

// 0x7C is Right Arrow; 0x7B Left Arrow
sendKey(0x7C)
