using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Timing : MonoBehaviour
{
    public float WaitTime;
    public GameObject LightningPSGroup;
    void Start()
    {
        StartCoroutine(Waiting());
    }

    IEnumerator Waiting()
    {
        print(Time.time);
        yield return new WaitForSeconds(WaitTime);
        LightningPSGroup.SetActive(true);
        print("Lightning Activated after" + Time.time + "seconds");
    }
}
