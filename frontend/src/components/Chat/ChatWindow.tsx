import React from 'react';
import { Message } from '../../types';
import MessageBubble from './MessageBubble';
import './ChatWindow.css';

interface Props {
  messages: Message[];
  loading: boolean;
}

const ChatWindow: React.FC<Props> = ({ messages, loading }) => {
  return (
    <div className="chat-window">
      {messages.map((msg, idx) => (
        <MessageBubble key={idx} message={msg} />
      ))}
      {loading && <div className="loading">...</div>}
    </div>
  );
};

export default ChatWindow; 