---
layout: default
title: SSH-NC Handbook
permalink: /ssh-nc-handbook/
---

# 🔐 SSH-NC Handbook: A Simple Guide to Remote Access & File Transfer

Whether you're experimenting with Kali Linux, Termux, or just want to learn how to control devices on your local network, **SSH** and **Netcat (NC)** are powerful, lightweight tools for remote login, terminal chat, and file sharing.

---

## ⚙️ What is SSH?

**SSH (Secure Shell)** allows you to remotely access another device’s terminal securely.

### 📌 Basic SSH Commands

```bash
# Connect to a remote device
ssh username@ip_address

# Add -v for verbose output (debug info)
ssh -v username@ip_address

# -n prevents reading from stdin (useful for scripting)
ssh -nv username@ip_address

# Copy files from local to remote
scp file.txt username@ip_address:/path/

# Copy files from remote to local
scp username@ip_address:/path/file.txt .
````

---

## 📂 How to Transfer Files Using SSH

### From Local to Remote

```bash
scp example.txt user@192.168.1.10:/home/user/
```

### From Remote to Local

```bash
scp user@192.168.1.10:/home/user/example.txt .
```

---

## 🌐 What is Netcat (NC)?

**Netcat** is a lightweight networking tool used to send/receive data over TCP or UDP.

### 🔄 Terminal Chat Over Local Network

**On Device A (Receiver):**

```bash
nc -lvp 1234
```

**On Device B (Sender):**

```bash
nc 192.168.1.10 1234
```

### 📁 Send a File

**On Receiver:**

```bash
nc -lvp 5555 > received.txt
```

**On Sender:**

```bash
nc 192.168.1.10 5555 < file.txt
```

---

## 📁 Send a Directory with Netcat (Compressed)

**Sender (Compress first):**

```bash
tar czf folder.tar.gz folder/
nc 192.168.1.10 4444 < folder.tar.gz
```

**Receiver:**

```bash
nc -lvp 4444 > folder.tar.gz
tar xzf folder.tar.gz
```

---

## 🧠 When to Use SSH vs Netcat

| Task                    | Use SSH | Use Netcat         |
| ----------------------- | ------- | ------------------ |
| Remote terminal login   | ✅       | ❌                  |
| File transfers (secure) | ✅       | ⚠️ (no encryption) |
| Chat/Terminal Pipe      | ❌       | ✅                  |
| Quick LAN testing       | ❌       | ✅                  |

---

## 🛠 Install Netcat on Termux

```bash
pkg update && pkg install netcat
```

Or for some versions:

```bash
pkg install busybox
busybox nc
```

---

## 📚 Wrap Up

SSH and Netcat are core tools every Linux user should master. With just a few commands, you can securely connect to devices, transfer files, or even set up chat over LAN.

If you're learning cybersecurity, DevOps, or networking — this handbook is for you.

---

🔗 [Back to Home](../index.md)
