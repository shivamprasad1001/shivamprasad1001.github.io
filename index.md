---
layout: default
title: Shivam Prasad | AI/ML Developer
description: Personal blog of Shivam Prasad, focusing on AI/ML development, web technologies, and cybersecurity tools
---

# Welcome to My Digital Workshop

## About Me

Hey there! I'm **Shivam Prasad**, an AI/ML Developer with a passion for creating intelligent solutions that solve real-world problems. I work at the intersection of data science, software engineering, and creative problem-solving to build AI systems that are both powerful and practical.

When I'm not training models or debugging neural networks, you can find me exploring web development frameworks or tinkering with cybersecurity tools. I believe in a holistic approach to technology—understanding the full stack makes for better specialized solutions.

## Technical Focus Areas

### 🤖 AI/ML Development

My primary focus is on machine learning and artificial intelligence, with expertise in:

- Large Language Models (LLMs) implementation and fine-tuning
- PyTorch for deep learning research and deployment
- LangChain for building context-aware AI applications
- Hugging Face ecosystem for NLP tasks and model sharing
- MLOps and model deployment pipelines

### 🌐 Web Development

I enjoy bringing AI solutions to life through accessible interfaces:

- Modern frontend frameworks (React, Vue)
- Backend systems that support AI inference
- RESTful and GraphQL APIs for model serving
- Performance optimization for ML in web environments

### 🔒 Cybersecurity

Security is a critical consideration in any AI deployment:

- Secure model deployment architectures
- Linux-based security tooling (SSH, Netcat)
- Mobile security testing with Termux
- Threat modeling for ML systems

## Recent Blog Posts

<div class="post-list">
  {% for post in site.posts limit:5 %}
    <div class="post-item">
      <h3><a href="{{ post.url }}">{{ post.title }}</a></h3>
      <span class="post-date">{{ post.date | date: "%B %d, %Y" }}</span>
      <p>{{ post.excerpt }}</p>
      <a href="{{ post.url }}" class="read-more">Read more →</a>
    </div>
  {% endfor %}
</div>

## Current Projects

I'm currently working on:

- Building a privacy-preserving recommendation system
- Exploring efficient fine-tuning methods for LLMs
- Contributing to open-source NLP tools
- Creating tutorials on secure AI deployment

## Terminal Toolbox

As someone who lives in the terminal, here are some of my favorite tools:

```bash
# Daily drivers
vim, tmux, git, docker

# Security toolkit
ssh, netcat, nmap, wireshark

# Mobile development
termux, adb, scrcpy

# ML workflow
jupyter, tensorboard, wandb
```

## Let's Connect

You can find my open-source contributions and projects on [GitHub](https://github.com/shivamprasad).

> "The best way to predict the future is to implement it." — Adapted from Alan Kay

Thanks for stopping by my digital workshop. Feel free to explore my projects and blog posts, or reach out if you're interested in collaborating on AI solutions that matter.
