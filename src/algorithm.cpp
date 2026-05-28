#include <iostream>

using namespace std;

// 结构体
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode* nxt): val(x), next(nxt) {}
};

// 辅助函数
void printList(ListNode* head) {
    while (head) {
        cout << head->val << " ";
        head = head->next;
    }
    cout << endl;
}

// 反转链表
ListNode* reverseList(ListNode* head) {
    ListNode* cur = head;
    ListNode* prev = nullptr;

    while (cur) {
        ListNode* temp = cur->next;
        cur->next = prev;
        prev = cur;
        cur = temp;
    }
    return prev;
}

// 从小到大合并链表
ListNode* mergeTwoList(ListNode* list1, ListNode* list2) {
    if (list1 == nullptr)
        return list2;

    else if (list2 == nullptr)
        return list1;

    else if (list1->val <= list2->val) {
        list1->next = mergeTwoList(list1->next, list2);
        return list1;
    }

    else {
        list2->next = mergeTwoList(list1, list2->next);
        return list2;
    }
}

// 环形链表
bool hasCycle(ListNode* head) {
    ListNode* slow = head;
    ListNode* fast = head;

    while (fast && fast->next) {
        fast = fast->next->next;
        slow = slow->next;
        if (slow == fast)
            return true;
    }
    return false;
}

// 移除倒数第n个节点
ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode* dummy = new ListNode(0, head);
    ListNode* slow = dummy;
    ListNode* fast = dummy;

    while (n--) {
        fast = fast->next;
    }

    while (fast->next) {
        slow = slow->next;
        fast = fast->next;
    }

    slow->next = slow->next->next;

    return dummy->next;
}
int main() {

    // cycle用例
    ListNode* head = new ListNode(1);
    head->next = new ListNode(2);
    head->next->next = new ListNode(3);
    head->next->next->next = new ListNode(4);

    ListNode* tmp = removeNthFromEnd(head, 1);

    printList(tmp);

    return 0;
}

