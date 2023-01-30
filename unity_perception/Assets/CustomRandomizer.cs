using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CustomRandomizer : MonoBehaviour
{
    public GameObject[] crates;

    // Start is called before the first frame update
    void Start()
    {
        foreach (var crate in crates)
        {
            var position = new Vector3(Random.Range(-2.0f, 2.0f), 0, Random.Range(-2.0f, 2.0f));
            crate.transform.position = position;
        }        
    }

    // Update is called once per frame
    void Update()
    {
        foreach (var crate in crates)
        {
            var position = new Vector3(Random.Range(-3.0f, 3.0f), 0, Random.Range(-3.0f, 3.0f));
            crate.transform.position = position;
        }        
    }
}
